"""Supplement joins must preserve identity, multiplicity, holdouts and evidence limits."""

import copy
import gzip
import json
import shutil
import sqlite3

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import sha256_file
from forge.corpus import compose_lipid_supplement as module


def component(identity="ethanol", structure="CCO"):
    return {"component_id": identity, "constitution": structure}


def test_catalogue_joins_global_structures_without_bare_alias_collision():
    values = [component(), component("amine", "CCN")]
    for row, family in zip(values, ("family_a", "family_b")):
        row["scoped_aliases"] = [
            {"family": family, "field": "head", "value": "A1", "scoped_alias": f"{family}|head|A1"}
        ]
    structures, report = module.catalogue(values)
    assert structures == {"ethanol": "CCO", "amine": "CCN"}
    assert report["ambiguous_scoped_aliases"] == 0


@pytest.mark.parametrize(
    "values",
    [
        [component(), component()],
        [component(), component("different_id")],
        [component(structure="OCC")],
        [component(structure="not-a-molecule")],
    ],
)
def test_catalogue_rejects_duplicate_or_noncanonical_structures(values):
    with pytest.raises(ComposeLipidError):
        module.catalogue(values)


@pytest.mark.parametrize("quantity", [True, 0, -1, 1.5, "2", None])
def test_quantity_must_be_a_positive_integer(quantity):
    with pytest.raises(ComposeLipidError):
        module.instances(
            [{"component_id": "x", "role": "tail", "quantity": quantity}],
            {"x": "CCO"},
            quantities=True,
        )


def test_quantity_reconciles_collapsed_export_and_repeated_manifest():
    one = {"component_id": "x", "role": "tail"}
    assert module.instances(
        [{**one, "quantity": 2}], {"x": "CCO"}, quantities=True
    ) == module.instances([one, one], {"x": "CCO"}, quantities=False)
    with pytest.raises(ComposeLipidError, match="Unresolved"):
        module.instances([one], {}, quantities=False)
    with pytest.raises(ComposeLipidError, match="disagreement"):
        module.instances([{**one, "constitution": "CCN"}], {"x": "CCO"}, quantities=False)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    module_path = tmp_path / "forge/corpus/compose_lipid_supplement.py"
    module_path.parent.mkdir(parents=True)
    shutil.copyfile(module.__file__, module_path)
    monkeypatch.setattr(module, "__file__", str(module_path))
    components = [component(), component("amine", "CCN")]
    records = []
    for target, family, identity, quantity in [
        ("train", "a", "ethanol", 2),
        ("test", "a", "amine", 1),
        ("outside", "b", "amine", 1),
    ]:
        records.append(
            {
                "target_id": target,
                "primary_family": family,
                "source_anchor": False,
                "construction_basis": "v8_exact_family_task",
                "family_definition_sufficient": True,
                "reaction_steps_sites": {},
                "provenance": {"task": "pinned-original-task"},
                "component_instances": [
                    {"component_id": identity, "role": "tail", "quantity": quantity}
                ],
                "constitution": "INVALID_PRODUCT_MUST_NOT_BE_PARSED",
            }
        )
    manifest = []
    groups = []
    assignments = []
    for record in records[:2]:
        instance = record["component_instances"][0]
        repeated = [
            {key: instance[key] for key in ("component_id", "role")}
            for _ in range(instance["quantity"])
        ]
        row = {
            "target_id": record["target_id"],
            "family": record["primary_family"],
            "component_instances": repeated,
            "component_ids": [instance["component_id"]],
            "source_pmids": [],
        }
        manifest.append(row)
        groups.append(copy.deepcopy(row))
        assignments.append(
            {
                "target_id": record["target_id"],
                "family": "a",
                "split": record["target_id"],
                "test_panels": (
                    ["unseen_component_structure"] if record["target_id"] == "test" else []
                ),
                "combination_signature": record["target_id"],
                "morphology_group_signature": record["target_id"],
            }
        )
    inputs = {}
    for name, values in {
        "precursors": components,
        "constructions": records,
        "components": manifest,
        "groups": groups,
        "assignments": assignments,
    }.items():
        path = tmp_path / f"{name}.gz"
        with gzip.open(path, "wt") as stream:
            for value in values:
                stream.write(json.dumps(value) + "\n")
        inputs[name] = path
    for name, value in {
        "selected_groups": {"selected_components": ["amine"], "selected_source_studies": []},
        "split_summary": {
            "split_counts": {"train": 1, "test": 1},
            "panel_counts": {"unseen_component_structure": 1},
        },
        "export_summary": {
            "precursor_structures": 2,
            "primary_family_counts": {"a": 2, "b": 1},
            "construction_basis_counts": {"v8_exact_family_task": 3},
        },
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        inputs[name] = path
    inputs["corpus"] = tmp_path / "original.sqlite"
    with sqlite3.connect(inputs["corpus"]) as db:
        db.execute(
            "CREATE TABLE targets(target_id TEXT PRIMARY KEY,family TEXT,source_anchor INTEGER)"
        )
        db.executemany(
            "INSERT INTO targets VALUES (?,?,0)", [("train", "a"), ("test", "a"), ("outside", "b")]
        )
        db.execute(
            "CREATE TABLE assignments(target_id TEXT PRIMARY KEY,family TEXT,forge_split TEXT)"
        )
        db.executemany(
            "INSERT INTO assignments VALUES (?,'a',?)", [("train", "train"), ("test", "heldout")]
        )
    config = {
        "inputs": {
            name: {"path": str(path.relative_to(tmp_path)), "sha256": str(sha256_file(path))}
            for name, path in inputs.items()
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return tmp_path, config_path, inputs, records


def test_complete_join_and_cross_family_protection_without_product_parsing(workspace):
    repo, config, paths, _ = workspace
    before = sha256_file(paths["corpus"])
    result = module.build_intake(repo, config, repo / "output")
    assert result["constructions"]["rows"] == 3
    assert result["constructions"]["quantity_expanded_references"] == 4
    assert not any(result["constructions"]["join_failures"].values())
    assert result["full_universe_selected_test_component_overlap"]["by_family"] == {"a": 1, "b": 1}
    assert result["policy"]["training_rows_admitted"] == 0
    assert result["policy"]["product_molecules_parsed"] == 0
    assert sha256_file(paths["corpus"]) == before


def test_compatible_replay_cannot_be_promoted_to_historical_route(workspace):
    _, _, _, records = workspace
    row = copy.deepcopy(records[0])
    row["construction_basis"] = "lnpdb_compatible_decomposition_replay"
    for claim in (None, True, 0):
        row["reaction_steps_sites"]["historical_route_claimed"] = claim
        with pytest.raises(ComposeLipidError, match="historical"):
            module.construction(row, {"ethanol": "CCO"}, 1)
    row["reaction_steps_sites"]["historical_route_claimed"] = False
    assert module.construction(row, {"ethanol": "CCO"}, 1)[5] == "false"


def test_changed_input_pin_fails_before_publication(workspace):
    repo, config, paths, _ = workspace
    paths["precursors"].write_bytes(b"changed")
    with pytest.raises((ComposeLipidError, ValueError)):
        module.build_intake(repo, config, repo / "output")
    assert not (repo / "output/result.json").exists()


def test_selected_component_leak_is_rejected(workspace):
    _, _, paths, _ = workspace
    paths["selected_groups"].write_text(
        json.dumps({"selected_components": ["ethanol"], "selected_source_studies": []})
    )
    with (
        sqlite3.connect(":memory:") as db,
        pytest.raises(ComposeLipidError, match="selected_component_leaks"),
    ):
        module.prepare_split(db, paths, {"ethanol": "CCO", "amine": "CCN"})


@pytest.mark.parametrize(
    "mutation",
    [
        "DELETE FROM targets WHERE target_id='outside'",
        "UPDATE targets SET family='wrong' WHERE target_id='outside'",
        "INSERT INTO targets VALUES ('missing_construction','b',0)",
    ],
)
def test_missing_extra_and_mismatched_target_joins_fail_closed(workspace, mutation):
    repo, config_path, paths, _ = workspace
    with sqlite3.connect(paths["corpus"]) as db:
        db.execute(mutation)
    config = json.loads(config_path.read_text())
    config["inputs"]["corpus"]["sha256"] = str(sha256_file(paths["corpus"]))
    config_path.write_text(json.dumps(config))
    with pytest.raises(ComposeLipidError, match="join checks failed"):
        module.build_intake(repo, config_path, repo / "output")
    assert not (repo / "output/result.json").exists()
