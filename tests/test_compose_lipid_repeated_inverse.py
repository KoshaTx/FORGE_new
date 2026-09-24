"""Program inference preserves missing metadata, alternate tuples and protected precursors."""

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
from forge.corpus import compose_lipid_repeated_inverse as module

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
    config = json.loads(
        (ROOT / "configs/multireaction/compose_lipid_v8_michael_program_v1.json").read_text()
    )
    source = json.loads((ROOT / config["inputs"]["adjudication"]["path"]).read_text())
    registry = json.loads((ROOT / config["inputs"]["registry"]["path"]).read_text())
    keys = json.loads((ROOT / config["inputs"]["decomposition_key"]["path"]).read_text())
    dump(tmp_path / "key.json", {family: keys[family] for family in config["families"]})
    (tmp_path / "paper.txt").write_text("unit-test primary asset fixture")
    assets = {
        "decomposition_key": pin(tmp_path, tmp_path / "key.json"),
        "supplement": pin(tmp_path, tmp_path / "paper.txt"),
    }
    registry["source_assets"] = assets
    for reaction in registry["reactions"]:
        for name in ("parent_registry", "functional_group_registry"):
            original = ROOT / reaction["implementation"][name]["path"]
            dest = tmp_path / original.name
            shutil.copyfile(original, dest)
            reaction["implementation"][name] = pin(tmp_path, dest)
    dump(tmp_path / "registry.json", registry)
    source["registry"] = pin(tmp_path, tmp_path / "registry.json")
    source["assets"] = assets
    dump(tmp_path / "source.json", source)
    controls = source["source_controls"]
    amide = "aza_michael_acrylamide"
    ester = "aza_michael_acrylate"

    def meta(label, count, interface):
        return {
            "head_id": label + "_head",
            "tail_id": label + "_tail",
            "occupancy": count,
            "interface_stratum": interface,
        }

    records = [
        ("prior_product", ester, "train", controls[0]["expected_product"], meta("p", 2, "O")),
        ("good_ester", ester, "train", controls[1]["expected_product"], meta("e", 2, "O")),
        ("good_amide", amide, "train", controls[2]["expected_product"], meta("n", 2, "N")),
        ("missing", ester, "train", "CCN", {"head_id": "unknown", "tail_id": "unknown"}),
        ("mixed", ester, "train", "CN(CCC(=O)OC)CCC(=O)OCC", meta("m", 2, "O")),
        ("ambiguous", ester, "train", "CN(CCC(=O)OC)CCC(=O)OCC", meta("a", 1, "O")),
        (
            "partial",
            amide,
            "train",
            source["ambiguity_controls"][0]["expected_product"],
            meta("partial", 5, "N"),
        ),
        ("heldout", ester, "heldout", "INVALID_UNPARSED_HOLDOUT", {}),
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
        [{"identity": digest(controls[2]["components"]["acceptor_tail"]), "fold": "heldout"}],
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
                    "target_id": "old_id",
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
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    protection.build_product_protection(tmp_path, Path("protect.json"), Path("protection"))
    config["inputs"] = {
        "registry": pin(tmp_path, tmp_path / "registry.json"),
        "adjudication": pin(tmp_path, tmp_path / "source.json"),
        "decomposition_key": pin(tmp_path, tmp_path / "key.json"),
        "protection_result": pin(tmp_path, tmp_path / "protection/result.json"),
    }
    dump(tmp_path / "config.json", config)
    return tmp_path


def test_complete_programs_exclusions_and_missing_occupancy_are_all_accounted(workspace):
    result = module.run_repeated_inverse(workspace, Path("config.json"), Path("out"))
    rows = {
        r["target_id"]: r
        for r in read_rows(workspace / result["artifacts"]["programs.jsonl.gz"]["path"])
    }
    assert set(rows) == {"good_ester", "good_amide", "missing", "mixed", "ambiguous", "partial"}
    assert rows["good_ester"]["eligible_after_known_exclusions"]
    assert rows["good_amide"]["computed_consistency_pass"]
    assert rows["good_amide"]["historical_protected_precursor"]
    assert not rows["good_amide"]["eligible_after_known_exclusions"]
    assert rows["missing"]["status"] == "excluded_missing_or_invalid_program_metadata"
    assert rows["mixed"]["status"] == "excluded_no_complete_inverse"
    assert rows["ambiguous"]["status"] == "excluded_ambiguous_complete_inverse"
    assert rows["partial"]["status"] == "excluded_forward_or_balance_failure"
    assert len(rows["partial"]["replay"]["forward_layers"][-1]) == 2
    assert all(not r["training_admitted"] for r in rows.values())
    assert not result["summary"]["training_ready"]
    assert module.verify_repeated_inverse(workspace, Path("out/result.json")) == result


@pytest.mark.parametrize("mutation", ["stale_registry_pin", "different_source_asset"])
def test_adjudication_cannot_mask_registry_source_asset(workspace, mutation):
    registry_path = workspace / "registry.json"
    registry = json.loads(registry_path.read_text())
    if mutation == "stale_registry_pin":
        registry["source_assets"]["supplement"]["sha256"] = "0" * 64
    else:
        (workspace / "other-paper.txt").write_text("another source")
        registry["source_assets"]["supplement"] = pin(workspace, workspace / "other-paper.txt")
    dump(registry_path, registry)
    config_path, source_path = workspace / "config.json", workspace / "source.json"
    config, source = json.loads(config_path.read_text()), json.loads(source_path.read_text())
    source["registry"] = config["inputs"]["registry"] = pin(workspace, registry_path)
    dump(source_path, source)
    config["inputs"]["adjudication"] = pin(workspace, source_path)
    dump(config_path, config)
    from forge.core.hashing import PinError

    with pytest.raises((ComposeLipidError, PinError), match="changed|source asset differ"):
        module.run_repeated_inverse(workspace, Path("config.json"), Path("out"))
    assert not (workspace / "out").exists()


@pytest.mark.parametrize("mutation", ["drop_row", "hide_ambiguity", "unprotect"])
def test_refreshed_hash_cannot_conceal_changed_program_or_protection(workspace, mutation):
    result = module.run_repeated_inverse(workspace, Path("config.json"), Path("out"))
    path = workspace / result["artifacts"]["programs.jsonl.gz"]["path"]
    rows = read_rows(path)
    if mutation == "drop_row":
        rows.pop()
    elif mutation == "hide_ambiguity":
        row = next(r for r in rows if r["target_id"] == "partial")
        row["replay"]["forward_layers"][-1] = row["replay"]["forward_layers"][-1][:1]
        row["computed_consistency_pass"] = True
    else:
        next(r for r in rows if r["target_id"] == "good_amide")[
            "historical_protected_precursor"
        ] = False
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    result["artifacts"]["programs.jsonl.gz"] = pin(workspace, path)
    dump(workspace / "out/result.json", result)
    with pytest.raises(ComposeLipidError, match="complete independent replay"):
        module.verify_repeated_inverse(workspace, Path("out/result.json"))


@pytest.mark.parametrize("mutation", ["drop_family", "wrong_interface", "shrink_events"])
def test_contract_cannot_drop_scope_or_change_interface(workspace, mutation):
    config = json.loads((workspace / "config.json").read_text())
    if mutation == "drop_family":
        config["families"].pop("aza_michael_acrylate")
    elif mutation == "wrong_interface":
        config["families"]["aza_michael_acrylate"]["interface_stratum"] = "N"
    else:
        config["search_bounds"]["maximum_events"] = 2
    dump(workspace / "config.json", config)
    with pytest.raises(ComposeLipidError):
        module.run_repeated_inverse(workspace, Path("config.json"), Path("out"))
    assert not (workspace / "out").exists()


def test_forged_training_readiness_is_rejected(workspace):
    result = module.run_repeated_inverse(workspace, Path("config.json"), Path("out"))
    result["summary"]["training_ready"] = True
    dump(workspace / "out/result.json", result)
    with pytest.raises(ComposeLipidError, match="complete independent replay"):
        module.verify_repeated_inverse(workspace, Path("out/result.json"))


def test_fresh_replay_is_identical_and_outputs_cannot_be_overwritten(workspace):
    first = module.run_repeated_inverse(workspace, Path("config.json"), Path("out"))
    second = module.run_repeated_inverse(workspace, Path("config.json"), Path("again"))
    assert (
        first["artifacts"]["programs.jsonl.gz"]["sha256"]
        == second["artifacts"]["programs.jsonl.gz"]["sha256"]
    )
    with pytest.raises(ComposeLipidError, match="fresh"):
        module.run_repeated_inverse(workspace, Path("config.json"), Path("out"))
