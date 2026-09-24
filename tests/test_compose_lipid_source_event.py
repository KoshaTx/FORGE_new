"""Enforced product/precursor exclusions and independent verification of source events."""

import gzip
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.assembly.families import RegistryAssemblyAdapter, constitutional_molecule
from forge.core.hashing import sha256_file
from forge.corpus import compose_lipid_protection as protection
from forge.corpus import compose_lipid_source_event as module

ROOT = Path(__file__).resolve().parents[1]


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
    family = "aldehyde_ugi3"
    source = json.loads(
        (ROOT / "results/phase1/compose_lipid_v8_ugi3_source_v1/adjudication.json").read_text()
    )
    for name in ("registry", "functional_group_registry"):
        path = tmp_path / (name + ".json")
        shutil.copyfile(ROOT / source[name]["path"], path)
        source[name] = pin(tmp_path, path)
    key = {family: {"roles": source["source_contract"]["roles"], "variable": []}}
    dump(tmp_path / "key.json", key)
    # Tiny primary-asset fixtures exercise attribution without copying large PDFs.
    (tmp_path / "source.txt").write_text("unit-test source asset")
    source["assets"] = {
        "decomposition_key": pin(tmp_path, tmp_path / "key.json"),
        "agile_si": pin(tmp_path, tmp_path / "source.txt"),
        "chen_si": pin(tmp_path, tmp_path / "source.txt"),
    }
    dump(tmp_path / "adjudication.json", source)
    controls = source["source_controls"]
    order = ["amine", "aldehyde", "isocyanide"]
    import_config = {
        "program_bindings": {family: {"precursor_id_order": order}},
        "historical_identity_guards": [
            {
                "input": "guards",
                "format": "json",
                "identity_kind": "constitutional_digest",
                "identity_field": "identity",
                "fold_fields": ["fold"],
            }
        ],
    }
    dump(tmp_path / "import_config.json", import_config)
    dump(
        tmp_path / "guards.json",
        [{"identity": digest(controls[1]["components"]["amine_head"]), "fold": "heldout"}],
    )
    dump(tmp_path / "catalogue.json", {family: {}})
    records = []
    for index, control in enumerate(controls):
        records.append(
            (
                control["label"],
                "train",
                control["expected_product"],
                {
                    "precursor_ids": [f"source_{index}_{r}" for r in order],
                    "design_lane": f"lane_{index}",
                },
            )
        )
    adapter = RegistryAssemblyAdapter.from_registry(
        tmp_path / "registry.json",
        reaction_id=source["reaction_id"],
        expected_sha256=source["registry"]["sha256"],
    )
    components = dict(zip(adapter.roles, ["CNC", "CC=O", "CCC[N+]#[C-]"], strict=True))
    (secondary,) = adapter.forward_products(components).products
    records.append(("secondary", "train", secondary, {"precursor_ids": ["s" + r for r in order]}))
    records.append(("heldout", "heldout", "INVALID_UNPARSED_HOLDOUT", {}))
    with sqlite3.connect(tmp_path / "corpus.sqlite") as db:
        db.execute("CREATE TABLE targets(target_id TEXT,constitution TEXT,payload TEXT)")
        db.execute("CREATE TABLE assignments(target_id TEXT,family TEXT,forge_split TEXT)")
        db.execute("CREATE TABLE train_checks(target_id TEXT,constitution_id TEXT)")
        for label, fold, smiles, meta in records:
            payload = {
                "target_id": label,
                "primary_family": family,
                "primary_metadata": meta,
                "constitution": smiles,
                "training_admissible": False,
            }
            db.execute("INSERT INTO targets VALUES(?,?,?)", (label, smiles, json.dumps(payload)))
            db.execute("INSERT INTO assignments VALUES(?,?,?)", (label, family, fold))
            if fold == "train":
                db.execute("INSERT INTO train_checks VALUES(?,?)", (label, digest(smiles)))
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
                    "target_id": "changed_provider_id",
                    "constitution": controls[0]["expected_product"],
                    "prior_development_split": "validation",
                }
            )
            + "\n"
        )
    protect_config = {
        "schema_version": protection.CONFIG_SCHEMA,
        "policy": protection.POLICY,
        "expected_families": [family],
        "inputs": {
            "import_result": pin(tmp_path, tmp_path / "import.json"),
            "prior_construction": pin(tmp_path, tmp_path / "prior.gz"),
        },
    }
    dump(tmp_path / "protection_config.json", protect_config)
    for name in set(module.IMPLEMENTATION) | set(protection.IMPLEMENTATION):
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    protection.build_product_protection(
        tmp_path, Path("protection_config.json"), Path("protection")
    )
    inputs = {
        "registry": source["registry"],
        "functional_group_registry": source["functional_group_registry"],
        "adjudication": pin(tmp_path, tmp_path / "adjudication.json"),
        "decomposition_key": pin(tmp_path, tmp_path / "key.json"),
        "protection_result": pin(tmp_path, tmp_path / "protection/result.json"),
    }
    config = {
        "schema_version": module.CONFIG_SCHEMA,
        "policy": module.POLICY,
        "family": family,
        "reaction_id": source["reaction_id"],
        "maximum_outcomes": 256,
        "inputs": inputs,
    }
    dump(tmp_path / "config.json", config)
    return tmp_path


def test_source_replay_enforces_both_protection_layers_and_preserves_lanes(workspace):
    result = module.run_source_event(workspace, Path("config.json"), Path("out"))
    rows = read_rows(workspace / result["artifacts"]["programs.jsonl.gz"]["path"])
    assert {r["target_id"] for r in rows} == {"AGILE_R6", "Chen_iso_A11B5C1", "secondary"}
    assert result["summary"]["computed_consistency_pass"] == 2
    assert result["summary"]["historical_protected_precursor"] == 1
    assert result["summary"]["eligible_after_known_exclusions"] == 1
    assert result["summary"]["check_failures"] == {"source_reactive_site_witness": 1}
    assert all(
        not r["training_admitted"] and not r["experimental_execution_admitted"] for r in rows
    )
    assert (
        next(r for r in rows if r["target_id"] == "AGILE_R6")["source_metadata"]["design_lane"]
        == "lane_1"
    )
    assert not result["summary"]["training_ready"]
    assert module.verify_source_event(workspace, Path("out/result.json")) == result


def test_forged_primary_site_witness_is_rejected_even_with_refreshed_hash(workspace):
    result = module.run_source_event(workspace, Path("config.json"), Path("out"))
    ledger = workspace / result["artifacts"]["programs.jsonl.gz"]["path"]
    rows = read_rows(ledger)
    bad = next(r for r in rows if r["target_id"] == "secondary")
    bad["checks"]["source_reactive_site_witness"] = True
    bad["computed_consistency_pass"] = True
    with gzip.open(ledger, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    result["artifacts"]["programs.jsonl.gz"] = pin(workspace, ledger)
    dump(workspace / "out/result.json", result)
    with pytest.raises(ComposeLipidError, match="independently replayed"):
        module.verify_source_event(workspace, Path("out/result.json"))


@pytest.mark.parametrize("defect", ["training_ready", "control", "missing_implementation"])
def test_forged_receipt_claims_are_rejected(workspace, defect):
    result = module.run_source_event(workspace, Path("config.json"), Path("out"))
    if defect == "training_ready":
        result["summary"]["training_ready"] = True
    elif defect == "control":
        result["source_controls"]["AGILE_R6"]["computed_consistency_pass"] = False
    else:
        result["implementation"].pop("forge/assembly/source_event.py")
    dump(workspace / "out/result.json", result)
    with pytest.raises(ComposeLipidError):
        module.verify_source_event(workspace, Path("out/result.json"))


@pytest.mark.parametrize("defect", ["role_order", "source_formula", "source_events"])
def test_incomplete_or_inconsistent_source_contract_cannot_run(workspace, defect):
    if defect == "role_order":
        # The fixture import verifier is stubbed; source-event validation still rejects this.
        path = workspace / "import_config.json"
        config = json.loads(path.read_text())
        config["program_bindings"]["aldehyde_ugi3"]["precursor_id_order"] = ["amine"]
        dump(path, config)
    else:
        path = workspace / "adjudication.json"
        source = json.loads(path.read_text())
        if defect == "source_formula":
            source["source_controls"][0]["expected_formula"] = "CH4"
        else:
            source["source_contract"]["events"] = 2
        dump(path, source)
        path = workspace / "config.json"
        config = json.loads(path.read_text())
        config["inputs"]["adjudication"] = pin(workspace, workspace / "adjudication.json")
        dump(path, config)
    with pytest.raises(ComposeLipidError):
        module.run_source_event(workspace, Path("config.json"), Path("out"))
    assert not (workspace / "out").exists()


def test_conflicting_component_label_excludes_every_associated_row_without_voting():
    rows = []
    for family, role, identity in [
        ("f", "head", "a"),
        ("f", "head", "b"),
        ("g", "head", "c"),
        ("f", "tail", "d"),
    ]:
        rows.append(
            {
                "family": family,
                "components": [{"role": role, "source_id": "opaque", "constitution_id": identity}],
                "computed_consistency_pass": True,
                "historical_protected_precursor": False,
            }
        )
    module.exclude_component_conflicts(rows)
    assert [r["eligible_after_known_exclusions"] for r in rows] == [False, False, True, True]


def test_outputs_are_immutable_and_replay_ledger_is_identical(workspace):
    first = module.run_source_event(workspace, Path("config.json"), Path("out"))
    with pytest.raises(ComposeLipidError, match="fresh"):
        module.run_source_event(workspace, Path("config.json"), Path("out"))
    second = module.run_source_event(workspace, Path("config.json"), Path("replay"))
    assert (
        first["artifacts"]["programs.jsonl.gz"]["sha256"]
        == second["artifacts"]["programs.jsonl.gz"]["sha256"]
    )
