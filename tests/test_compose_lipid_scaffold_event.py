"""Whole-corpus scaffold-event receipts retain every exclusion and resist rehashed edits."""

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
from forge.corpus import compose_lipid_scaffold_event as module

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value))


def pin(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": str(sha256_file(path))}


def digest(smiles):
    return hashlib.sha256(constitutional_molecule(smiles)[0].encode()).hexdigest()


def rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    config = json.loads(
        (ROOT / "configs/multireaction/compose_lipid_v8_reductive_program_v1.json").read_text()
    )
    source = json.loads((ROOT / config["inputs"]["adjudication"]["path"]).read_text())
    registry = json.loads((ROOT / config["inputs"]["registry"]["path"]).read_text())
    keys = json.loads((ROOT / config["inputs"]["decomposition_key"]["path"]).read_text())
    dump(tmp_path / "key.json", {family: keys[family] for family in config["families"]})
    assets = {}
    for name, original in source["assets"].items():
        target = tmp_path / ("key.json" if name == "decomposition_key" else name)
        if name in ("scaffolds.json", "controls.json"):
            shutil.copyfile(ROOT / original["path"], target)
        elif name != "decomposition_key":
            target.write_text("unit-test primary asset bytes")
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
    controls = json.loads((tmp_path / "controls.json").read_text())["controls"]
    aliphatic, aryl = "reductive_amination", "aryl_reductive_amination"

    def meta(label):
        return {"head_id": label + "_head", "aldehyde_id": label + "_aldehyde"}

    records = [
        ("prior_product", aliphatic, "train", controls[0]["expected_product"], meta("p")),
        ("good_aliphatic", aliphatic, "train", controls[1]["expected_product"], meta("al")),
        ("good_aryl", aryl, "train", controls[2]["expected_product"], meta("ar")),
        (
            "unequal",
            aryl,
            "train",
            controls[2]["expected_product"].replace("OC(=O)C", "OC(=O)CC", 1),
            meta("u"),
        ),
        ("missing", aryl, "train", "CCN", {}),
        ("wrong_scaffold", aryl, "train", "CN(C)CCNCCCC", meta("w")),
        (
            "ambiguous",
            aryl,
            "train",
            "CCNCCCNCc1cc(OC(=O)C(CC)CCCC)cc(OC(=O)C(CC)CCCC)c1",
            meta("a"),
        ),
        ("heldout", aryl, "heldout", "INVALID_UNPARSED_HOLDOUT", {}),
    ]
    with sqlite3.connect(tmp_path / "corpus.sqlite") as db:
        db.execute("CREATE TABLE targets(target_id TEXT,constitution TEXT,payload TEXT)")
        db.execute("CREATE TABLE assignments(target_id TEXT,family TEXT,forge_split TEXT)")
        db.execute("CREATE TABLE train_checks(target_id TEXT,constitution_id TEXT)")
        for label, family, fold, smiles, metadata in records:
            payload = {
                "target_id": label,
                "primary_family": family,
                "primary_metadata": metadata,
                "constitution": smiles,
                "training_admissible": False,
            }
            db.execute("INSERT INTO targets VALUES(?,?,?)", (label, smiles, json.dumps(payload)))
            db.execute("INSERT INTO assignments VALUES(?,?,?)", (label, family, fold))
            if fold == "train":
                db.execute("INSERT INTO train_checks VALUES(?,?)", (label, digest(smiles)))
    dump(
        tmp_path / "guards.json",
        [{"identity": digest(controls[1]["components"]["amine_head"]), "fold": "heldout"}],
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
    dump(tmp_path / "catalogue.json", {family: {} for family in config["families"]})
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
    monkeypatch.setattr(protection, "verify_compose_lipid", lambda *_: imported)
    with gzip.open(tmp_path / "prior.gz", "wt") as stream:
        stream.write(
            json.dumps(
                {
                    "target_id": "earlier",
                    "constitution": controls[0]["expected_product"],
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
            "expected_families": list(config["families"]),
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


def test_all_rows_and_source_invariants_are_replayed_with_protection(workspace):
    result = module.run_scaffold_event(workspace, Path("config.json"), Path("out"))
    ledger = {
        r["target_id"]: r
        for r in rows(workspace / result["artifacts"]["programs.jsonl.gz"]["path"])
    }
    assert set(ledger) == {
        "good_aliphatic",
        "good_aryl",
        "unequal",
        "missing",
        "wrong_scaffold",
        "ambiguous",
    }
    assert ledger["good_aryl"]["eligible_after_known_exclusions"]
    assert ledger["good_aryl"]["computed_architecture_subfamily"] == "xue_a2_aryl_core"
    assert ledger["good_aliphatic"]["computed_consistency_pass"]
    assert ledger["good_aliphatic"]["historical_protected_precursor"]
    assert not ledger["good_aliphatic"]["eligible_after_known_exclusions"]
    assert ledger["unequal"]["status"] == "excluded_precursor_scaffold"
    assert ledger["missing"]["status"] == "excluded_missing_component_labels"
    assert ledger["wrong_scaffold"]["status"] == "excluded_no_inverse"
    assert ledger["ambiguous"]["status"] == "excluded_forward_or_stoichiometry"
    assert all(not r["training_admitted"] for r in ledger.values())
    assert module.verify_scaffold_event(workspace, Path("out/result.json")) == result


@pytest.mark.parametrize(
    "mutation", ["drop_row", "conceal_unequal_arms", "unprotect", "claim_ready"]
)
def test_refreshed_hash_does_not_admit_forged_scientific_result(workspace, mutation):
    result = module.run_scaffold_event(workspace, Path("config.json"), Path("out"))
    path = workspace / result["artifacts"]["programs.jsonl.gz"]["path"]
    records = rows(path)
    if mutation == "drop_row":
        records.pop()
    elif mutation == "claim_ready":
        result["summary"]["training_ready"] = True
    else:
        row = next(
            r
            for r in records
            if r["target_id"]
            == ("unequal" if mutation == "conceal_unequal_arms" else "good_aliphatic")
        )
        row["computed_consistency_pass"] = True
        row["eligible_after_known_exclusions"] = True
        row["historical_protected_precursor"] = False
    with gzip.open(path, "wt") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    result["artifacts"]["programs.jsonl.gz"] = pin(workspace, path)
    dump(workspace / "out/result.json", result)
    with pytest.raises(ComposeLipidError, match="independent replay"):
        module.verify_scaffold_event(workspace, Path("out/result.json"))


@pytest.mark.parametrize("mutation", ["drop_family", "reverse_roles"])
def test_scope_and_source_role_changes_fail_before_output(workspace, mutation):
    path = workspace / "config.json"
    config = json.loads(path.read_text())
    if mutation == "drop_family":
        config["families"].pop("reductive_amination")
    else:
        fields = config["families"]["reductive_amination"]["role_fields"]
        fields["amine_head"], fields["coupled_aldehyde"] = (
            fields["coupled_aldehyde"],
            fields["amine_head"],
        )
    dump(path, config)
    with pytest.raises(ComposeLipidError, match="scope|contract"):
        module.run_scaffold_event(workspace, Path("config.json"), Path("out"))
    assert not (workspace / "out").exists()


def test_ledger_reproduction_and_completed_output_immutability(workspace):
    a = module.run_scaffold_event(workspace, Path("config.json"), Path("a"))
    b = module.run_scaffold_event(workspace, Path("config.json"), Path("b"))
    assert (
        a["artifacts"]["programs.jsonl.gz"]["sha256"]
        == b["artifacts"]["programs.jsonl.gz"]["sha256"]
    )
    with pytest.raises(ComposeLipidError, match="fresh"):
        module.run_scaffold_event(workspace, Path("config.json"), Path("a"))
