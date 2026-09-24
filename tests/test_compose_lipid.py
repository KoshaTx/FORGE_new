from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError, program_catalogue, role_metadata
from forge.core.hashing import PinError, sha256_file
from forge.corpus.compose_lipid import (
    ComposeLipidCorpus,
    import_compose_lipid,
    verify_compose_lipid,
)

REPO = Path(__file__).resolve().parents[1]


def _dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def _jsonl(path, rows):
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def release(tmp_path):
    # These deliberately opaque IDs are not hashes of SMILES. Provider IDs remain join keys.
    molecules = ["CC", "CCC", "CCCC", "CCCCC", "C" * 81, "CCN", "NCCO", "CC(O)C"]
    families = ["first"] * 5 + ["second"] * 2 + ["reference"]
    folds = ["train", "train", "test", None, "train", "train", "calibration", "reference"]
    rows = []
    assignments = []
    for index, (smiles, family, fold) in enumerate(zip(molecules, families, folds, strict=True)):
        target = hashlib.sha256(f"provider:{index}".encode()).hexdigest()
        rows.append(
            {
                "target_id": target,
                "constitution": smiles,
                "primary_family": family,
                "primary_metadata": {"head_id": f"opaque:{index}", "occupancy": 3},
                "route_families": [family],
                "source_anchor": family == "reference",
                "heavy_atoms": len(smiles) if "(" not in smiles else 4,
                "size_disposition": "above_80_heavy_atom_hold" if index == 4 else "model_supported",
                "training_admissible": False,
            }
        )
        if fold:
            assignments.append(
                {
                    "target_id": target,
                    "family": family,
                    "split": fold,
                    "test_panels": ["unseen_exact_combination"] if fold == "test" else [],
                    **{
                        field: str(index)
                        for field in (
                            "combination_signature",
                            "core_scaffold_signature",
                            "regional_topology_signature",
                            "structural_group_signature",
                        )
                    },
                }
            )
    accepted = [
        dict(row, admission_lane="provider_lane")
        for row, fold in zip(rows, folds, strict=True)
        if fold
    ]
    key = {
        family: {
            "roles": {"head": 1},
            "field_to_role": {"head_id": ["head"]},
            "variable": ["occupancy"],
            "architecture_subfamilies": [{"id": "source_defined"}],
            "reaction_program_invariant": "fixture metadata only",
            "literature_libraries": [],
        }
        for family in ("first", "second")
    }
    _jsonl(tmp_path / "global_candidates.jsonl.gz", rows)
    _jsonl(tmp_path / "accepted_targets.jsonl.gz", accepted)
    _jsonl(tmp_path / "assignments.jsonl.gz", assignments)
    _dump(tmp_path / "key.json", key)
    _dump(tmp_path / "registry.json", {"reactions": [{"reaction_id": "existing"}]})
    (tmp_path / "guards.csv").write_text("smiles,fold\nCCC,heldout\n")
    shutil.copyfile(
        REPO / "results/phase1/product_v3_atom_vocabulary.json", tmp_path / "vocabulary.json"
    )
    names = {
        "universe": "global_candidates.jsonl.gz",
        "accepted": "accepted_targets.jsonl.gz",
        "assignments": "assignments.jsonl.gz",
        "decomposition_key": "key.json",
        "registry": "registry.json",
        "guard": "guards.csv",
        "atom_vocabulary": "vocabulary.json",
    }
    (tmp_path / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(tmp_path / names[k])}  {names[k]}\n"
            for k in ("universe", "accepted", "assignments")
        )
    )
    names["checksums"] = "SHA256SUMS"
    config = {
        "schema_version": "forge.compose_lipid_import_config.v1",
        "inputs": {
            k: {"path": name, "sha256": str(sha256_file(tmp_path / name))}
            for k, name in names.items()
        },
        "registries": ["registry"],
        "reference_families": ["reference"],
        "program_bindings": {
            f: {"precursor_id_order": [], "related_registry_reactions": ["existing"]} for f in key
        },
        "historical_identity_guards": [
            {"input": "guard", "identity_field": "smiles", "fold_fields": ["fold"]}
        ],
        "admission_policy": "preserve_source_flags_require_exact_program",
        "sampling_policy": "equal_family_mass_after_qualification",
        "graph_probes_per_family": 1,
        "expected": {
            "universe_by_family": dict(Counter(families)),
            "accepted_by_family": dict(Counter(row["primary_family"] for row in accepted)),
            "provider_splits": dict(Counter(row["split"] for row in assignments)),
        },
    }
    _dump(tmp_path / "config.json", config)
    for name in (
        "forge/corpus/compose_lipid.py",
        "forge/assembly/compose_lipid.py",
        "forge/corpus/library_splits.py",
        "forge/assembly/families.py",
        "forge/corpus/combinatorial_libraries.py",
        "forge/model/sparse_topology_feasibility.py",
        "forge/model/vocabulary.py",
        "forge/core/hashing.py",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    return tmp_path, config, rows, accepted, assignments


def _repin(repo, config, name):
    config["inputs"][name]["sha256"] = str(sha256_file(repo / config["inputs"][name]["path"]))
    checksums = repo / "SHA256SUMS"
    checksums.write_text(
        "".join(
            f"{config['inputs'][k]['sha256']}  {config['inputs'][k]['path']}\n"
            for k in ("universe", "accepted", "assignments")
        )
    )
    config["inputs"]["checksums"]["sha256"] = str(sha256_file(checksums))
    _dump(repo / "config.json", config)


def test_import_preserves_universe_splits_large_graphs_and_source_flags(release):
    repo, _, rows, _, _ = release
    result = import_compose_lipid(repo, Path("config.json"), Path("run"))
    summary = result["summary"]
    assert summary["universe_rows"] == 8
    assert summary["accepted_rows"] == summary["assignment_rows"] == 7
    assert summary["forge_splits"] == {
        "train": 3,
        "quarantine": 1,
        "heldout": 1,
        "calibration": 1,
        "reference": 1,
    }
    assert summary["train_quarantine_reasons"] == {"historical_identity_holdout": 1}
    assert summary["size_dispositions"]["above_80_heavy_atom_hold"] == 1
    assert summary["exact_program_qualified_training_rows"] == 0
    corpus = ComposeLipidCorpus(repo, Path("run/result.json"))
    train = list(corpus.iter_records(split="train"))
    assert len(train) == 3
    assert any(row["source"]["heavy_atoms"] == 81 for row in train)
    assert all(row["source"]["training_admissible"] is False for row in train)
    assert all(row["training_sampling_weight"] == 0 for row in train)
    assert list(corpus.iter_records(split="unassigned"))[0]["source"] == rows[3]
    quarantined = list(corpus.iter_records(split="quarantine"))[0]
    assert quarantined["assignment"]["split"] == "train"
    assert quarantined["quarantine_reason"] == "historical_identity_holdout"
    assert list(corpus.iter_records(split="heldout"))[0]["role_metadata"] is None
    with pytest.raises(ComposeLipidError, match="not training admission"):
        corpus.iter_training_records()
    with pytest.raises(ComposeLipidError, match="explicit"):
        list(corpus.iter_records(split="all"))
    with pytest.raises(ComposeLipidError, match="unknown family"):
        list(corpus.iter_records(split="train", family="absent"))
    assert verify_compose_lipid(repo, Path("run/result.json"))["summary"] == summary


def test_repeat_import_deterministic_artifacts_and_refuses_overwrite(release):
    repo, *_ = release
    a = import_compose_lipid(repo, Path("config.json"), Path("a"))
    b = import_compose_lipid(repo, Path("config.json"), Path("b"))
    assert a["summary"] == b["summary"]
    assert a["graph_probes"] == b["graph_probes"]
    for name in a["artifacts"]:
        assert a["artifacts"][name]["sha256"] == b["artifacts"][name]["sha256"]
    with pytest.raises(ComposeLipidError, match="fresh"):
        import_compose_lipid(repo, Path("config.json"), Path("a"))


def test_provider_duplicate_graph_ids_preserved_but_training_quarantined(release):
    repo, config, rows, accepted, assignments = release
    # A held-out graph with a different provider ID must not re-enter the TRAIN view.
    rows[2]["constitution"] = rows[0]["constitution"]
    rows[2]["heavy_atoms"] = rows[0]["heavy_atoms"]
    accepted[2].update(rows[2])
    _jsonl(repo / "global_candidates.jsonl.gz", rows)
    _jsonl(repo / "accepted_targets.jsonl.gz", accepted)
    _repin(repo, config, "universe")
    _repin(repo, config, "accepted")
    result = import_compose_lipid(repo, Path("config.json"), Path("duplicates"))
    assert result["summary"]["universe_rows"] == 8
    assert result["summary"]["distinct_source_constitutions"] == 7
    assert result["summary"]["duplicate_constitution_classes"] == 1
    assert result["summary"]["train_quarantine_reasons"]["duplicate_source_constitution"] == 1
    corpus = ComposeLipidCorpus(repo, Path("duplicates/result.json"))
    assert rows[0]["target_id"] not in {
        r["source"]["target_id"] for r in corpus.iter_records(split="train")
    }
    assert list(corpus.iter_records(split="heldout"))[0]["assignment"] == assignments[2]


def test_large_anchor_provider_size_label_does_not_drop_or_admit_graph(release):
    repo, config, rows, accepted, _ = release
    rows[4].update(source_anchor=True, size_disposition="model_supported")
    accepted[3].update(rows[4])
    _jsonl(repo / "global_candidates.jsonl.gz", rows)
    _jsonl(repo / "accepted_targets.jsonl.gz", accepted)
    _repin(repo, config, "universe")
    _repin(repo, config, "accepted")
    result = import_compose_lipid(repo, Path("config.json"), Path("anchors"))
    assert result["summary"]["source_size_label_disagreements_with_readme"] == 1
    assert result["summary"]["above_80_atoms_by_source_count"] == 1
    assert not result["admission"]["training_available"]


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "unknown_family",
        "bad_join",
        "missing_assignment",
        "changed_split",
        "changed_population",
        "raw_weighting",
        "unknown_adapter",
    ],
)
def test_invalid_release_never_publishes_partial_index(release, defect):
    repo, config, rows, accepted, assignments = release
    if defect == "duplicate":
        rows.append(rows[0])
        _jsonl(repo / "global_candidates.jsonl.gz", rows)
        _repin(repo, config, "universe")
    elif defect == "unknown_family":
        rows[0]["primary_family"] = "unregistered"
        _jsonl(repo / "global_candidates.jsonl.gz", rows)
        _repin(repo, config, "universe")
    elif defect == "bad_join":
        accepted[0]["constitution"] = "CN"
        _jsonl(repo / "accepted_targets.jsonl.gz", accepted)
        _repin(repo, config, "accepted")
    elif defect == "missing_assignment":
        _jsonl(repo / "assignments.jsonl.gz", assignments[1:])
        _repin(repo, config, "assignments")
    elif defect == "changed_split":
        assignments[0]["split"] = "reference"
        _jsonl(repo / "assignments.jsonl.gz", assignments)
        _repin(repo, config, "assignments")
    else:
        if defect == "changed_population":
            config["expected"]["provider_splits"]["train"] += 1
        elif defect == "raw_weighting":
            config["sampling_policy"] = "raw_family_count"
        else:
            config["program_bindings"]["first"]["related_registry_reactions"] = ["invented"]
        _dump(repo / "config.json", config)
    with pytest.raises(ComposeLipidError):
        import_compose_lipid(repo, Path("config.json"), Path("bad"))
    assert not (repo / "bad").exists()


def test_tampered_input_and_published_index_are_rejected(release):
    repo, config, *_ = release
    import_compose_lipid(repo, Path("config.json"), Path("good"))
    with sqlite3.connect(repo / "good/corpus.sqlite") as db:
        db.execute("UPDATE assignments SET forge_split='train' WHERE forge_split='heldout'")
    with pytest.raises(PinError):
        ComposeLipidCorpus(repo, Path("good/result.json"))
    config["inputs"]["accepted"]["sha256"] = "0" * 64
    _dump(repo / "config.json", config)
    with pytest.raises(PinError):
        import_compose_lipid(repo, Path("config.json"), Path("bad"))
    assert not (repo / "bad").exists()


def test_roles_preserve_order_coupling_and_unresolved_anchor_metadata():
    key = {
        "example": {"roles": {"head": 1, "tail": 2}, "field_to_role": {}, "variable": ["events"]}
    }
    catalogue = program_catalogue(
        key,
        {"example": {"precursor_id_order": ["tail", "head"], "related_registry_reactions": []}},
        set(),
    )
    definition = catalogue["example"]
    metadata = role_metadata(
        {"target_id": "opaque", "primary_metadata": {"precursor_ids": ["T", "H"], "events": 3}},
        definition,
    )
    assert metadata["roles"]["tail"] == {
        "source_nominal_count": 2,
        "metadata": {"precursor_id": "T"},
    }
    assert metadata["roles"]["head"]["metadata"]["precursor_id"] == "H"
    assert metadata["variable_multiplicity_metadata"] == {"events": 3}
    assert not metadata["exact_atom_partition_verified"]
    anchor = role_metadata(
        {"target_id": "anchor", "primary_metadata": {"source_release": "paper"}}, definition
    )
    assert anchor["missing_role_metadata"] == ["head", "tail"]
    with pytest.raises(ComposeLipidError, match="arity"):
        role_metadata(
            {"target_id": "bad", "primary_metadata": {"precursor_ids": ["T"]}}, definition
        )


def test_cli_import_and_verify(release, monkeypatch, capsys):
    import cli

    repo, *_ = release
    monkeypatch.setattr(cli, "_repo", lambda: repo)
    assert (
        cli.main(
            ["data", "compose-lipid", "import", "--config", "config.json", "--output", "cli-run"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["summary"]["universe_rows"] == 8
    assert cli.main(["data", "compose-lipid", "verify", "--result", "cli-run/result.json"]) == 0
