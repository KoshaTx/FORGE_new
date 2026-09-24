"""Global identity comparisons cannot turn unresolved source labels into held-out chemistry."""

import gzip
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import PinError, sha256_file
from forge.corpus import compose_lipid_precursor_audit as module
from forge.corpus.library_splits import FrozenIdentityFolds

ROOT = Path(__file__).resolve().parents[1]


def digest(smiles):
    return hashlib.sha256(smiles.encode()).hexdigest()


def catalogue():
    return {
        family: {
            "source_definition": {
                "roles": {"head": 1},
                "field_to_role": {"head_id": ["head"]},
                "variable": [],
            },
            "precursor_id_order": [],
        }
        for family in ("a", "b", "missing")
    }


def source(target, family, label, *, eligible=True):
    return {
        "target_id": target,
        "family": family,
        "constitution_id": digest(target),
        "eligible_for_program_preparation": eligible,
        "primary_metadata": {"head_id": label},
    }


def evidence(target, family, label, smiles, *, fold=None):
    return {
        "target_id": target,
        "family": family,
        "exact_program": True,
        "computed_consistency_pass": True,
        "training_admitted": False,
        "historical_protected_precursor": fold in {"calibration", "heldout"},
        "components": [
            {
                "role": "head",
                "source_id": label,
                "canonical_smiles": smiles,
                "constitution_id": digest(smiles),
                "historical_fold": fold,
            }
        ],
    }


def panel(target, family, label, *, name="unseen_precursor_identity"):
    return {
        "target_id": target,
        "family": family,
        "split": "heldout",
        "primary_metadata": {"head_id": label},
        "test_panels": [name],
        "constitution": "MUST_NEVER_BE_PARSED",
    }


def population():
    rows = [source("a1", "a", "head-a"), source("b1", "b", "head-b"), source("m1", "missing", "m")]
    evidence_rows = [evidence("a1", "a", "head-a", "CCN"), evidence("b1", "b", "head-b", "CCN")]
    return {r["target_id"]: r for r in rows}, evidence_rows


def run(preparation=None, rows=None, panels=(), guards=None):
    initial, initial_rows = population()
    return module.audit_precursors(
        initial if preparation is None else preparation,
        initial_rows if rows is None else rows,
        panels,
        catalogue(),
        guards or FrozenIdentityFolds(),
    )


def test_cross_family_aliases_collapse_by_chemistry_without_training_admission():
    result = run()
    assert result["summary"]["unique_exact_precursor_identities"] == 1
    assert result["summary"]["identities_shared_across_families"] == 1
    assert result["identities"][0]["families"] == ["a", "b"]
    assert result["summary"]["totals"]["preparation_rows_without_program_evidence"] == 1
    assert not result["summary"]["training_ready"]
    assert all(row["training_admitted"] is False for row in result["rows"])


def test_known_historical_protection_applies_to_every_role_alias():
    preparation, rows = population()
    guards = FrozenIdentityFolds()
    guards.add(digest("CCN"), "heldout", "historical")
    for row in rows:
        row["components"][0]["historical_fold"] = "heldout"
        row["historical_protected_precursor"] = True
    result = run(preparation, rows, guards=guards)
    assert result["summary"]["totals"]["exact_rows_with_known_protected_precursor"] == 2
    assert not any(r["clear_of_known_exclusions"] for r in result["rows"])


def test_unknown_evaluation_label_is_unresolved_not_novel_and_does_not_fit_dictionary():
    before = run()
    result = run(panels=[panel("eval", "a", "unknown")])
    assert result["identities"] == before["identities"]
    assert result["rows"] == before["rows"]
    checked = result["panels"][0]
    assert checked["unresolved_roles"] == ["head"]
    assert not checked["all_roles_resolved_from_preparation"]
    assert not checked["chemical_identity_holdout_qualified"]


def test_label_implied_overlap_is_reported_only_for_unseen_precursor_panel():
    result = run(
        panels=[
            panel("identity", "a", "head-a"),
            panel("combination", "a", "head-a", name="unseen_exact_combination"),
        ]
    )
    rows = {r["target_id"]: r for r in result["panels"]}
    assert rows["identity"]["unseen_precursor_label_claim_contradicted"]
    assert not rows["combination"]["unseen_precursor_label_claim_contradicted"]
    assert rows["combination"]["all_role_identities_present_in_clear_exact_rows"]


def test_labels_are_scoped_by_family_not_assumed_global_hashes():
    preparation, rows = population()
    rows[1] = evidence("b1", "b", "head-b", "CCO")
    result = run(preparation, rows, [panel("eval", "b", "head-a")])
    assert result["panels"][0]["unresolved_roles"] == ["head"]
    assert result["summary"]["conflicting_scoped_labels"] == 0


def test_conflicting_scoped_label_excludes_every_affected_row_and_abstains_on_panel():
    preparation, rows = population()
    preparation["a2"] = source("a2", "a", "head-a")
    rows.append(evidence("a2", "a", "head-a", "CCO"))
    result = run(preparation, rows, [panel("eval", "a", "head-a")])
    assert result["summary"]["conflicting_scoped_labels"] == 1
    assert result["summary"]["by_family"]["a"]["exact_rows_with_conflicting_labels"] == 2
    assert result["panels"][0]["unresolved_roles"] == ["head"]
    assert all(not r["clear_of_known_exclusions"] for r in result["rows"] if r["family"] == "a")


def test_prior_product_exclusions_cannot_fit_component_dictionary():
    preparation, rows = population()
    preparation["a1"]["eligible_for_program_preparation"] = False
    result = run(preparation, rows, [panel("eval", "a", "head-a")])
    assert result["panels"][0]["unresolved_roles"] == ["head"]
    assert result["identities"][0]["families"] == ["b"]
    assert result["summary"]["by_family"]["a"]["program_rows_excluded_by_prior_product"] == 1


@pytest.mark.parametrize(
    "defect",
    [
        "digest",
        "canonical",
        "source_label",
        "role",
        "fold",
        "protection",
        "admission",
        "conflict",
        "nonboolean",
        "duplicate",
        "nontrain",
        "family",
    ],
)
def test_invalid_program_evidence_fails_closed(defect):
    preparation, rows = population()
    c = rows[0]["components"][0]
    if defect == "digest":
        c["constitution_id"] = "0" * 64
    elif defect == "canonical":
        c["canonical_smiles"] = "NCC"
    elif defect == "source_label":
        c["source_id"] = "other"
    elif defect == "role":
        c["role"] = "tail"
    elif defect == "fold":
        c["historical_fold"] = "heldout"
    elif defect == "protection":
        rows[0]["historical_protected_precursor"] = True
    elif defect == "admission":
        rows[0]["training_admitted"] = True
    elif defect == "conflict":
        rows[0]["component_label_conflict"] = True
    elif defect == "nonboolean":
        rows[0]["exact_program"] = 1
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "nontrain":
        rows[0]["target_id"] = "eval"
    else:
        rows[0]["family"] = "b"
    with pytest.raises(ComposeLipidError):
        run(preparation, rows)


def test_missing_family_cannot_disappear_from_scope():
    preparation, rows = population()
    del preparation["m1"]
    with pytest.raises(ComposeLipidError, match="every imported family"):
        run(preparation, rows)


def test_malformed_eval_metadata_is_retained_without_a_success_claim():
    row = panel("eval", "a", "id")
    row["primary_metadata"] = {}
    result = run(panels=[row])
    assert result["panels"][0]["malformed_metadata"]
    assert not result["panels"][0]["all_roles_resolved_from_preparation"]


def test_order_does_not_affect_any_scientific_output():
    preparation, rows = population()
    panels = [panel("x", "a", "head-a"), panel("y", "b", "head-b")]
    assert run(preparation, rows, panels) == run(
        dict(reversed(list(preparation.items()))), list(reversed(rows)), list(reversed(panels))
    )


def dump(path, value):
    path.write_text(json.dumps(value))


def pin(repo, path):
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def write_rows(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    preparation, rows = population()
    write_rows(tmp_path / "preparation.gz", preparation.values())
    write_rows(tmp_path / "programs.gz", rows)
    dump(tmp_path / "catalogue.json", catalogue())
    dump(tmp_path / "import_config.json", {"historical_identity_guards": []})
    dump(tmp_path / "program_config.json", {})
    with sqlite3.connect(tmp_path / "corpus.sqlite") as db:
        db.execute(
            "CREATE TABLE assignments(target_id TEXT, family TEXT, forge_split TEXT, payload TEXT)"
        )
        db.execute("CREATE TABLE targets(target_id TEXT, payload TEXT)")
        for row in [*preparation.values(), panel("eval", "a", "head-a")]:
            db.execute(
                "INSERT INTO targets VALUES (?,?)",
                (
                    row["target_id"],
                    json.dumps(
                        {
                            "primary_metadata": row["primary_metadata"],
                            "constitution": "NEVER_PARSE_THIS",
                        }
                    ),
                ),
            )
            db.execute(
                "INSERT INTO assignments VALUES (?,?,?,?)",
                (
                    row["target_id"],
                    row["family"],
                    row.get("split", "train"),
                    json.dumps({"test_panels": row.get("test_panels", [])}),
                ),
            )
    imported = {
        "config": pin(tmp_path, tmp_path / "import_config.json"),
        "inputs": {},
        "artifacts": {
            "corpus.sqlite": pin(tmp_path, tmp_path / "corpus.sqlite"),
            "program_catalogue.json": pin(tmp_path, tmp_path / "catalogue.json"),
        },
    }
    dump(tmp_path / "import.json", imported)
    protection = {
        "inputs": {"import_result": pin(tmp_path, tmp_path / "import.json")},
        "artifacts": {"preparation_view.jsonl.gz": pin(tmp_path, tmp_path / "preparation.gz")},
    }
    dump(tmp_path / "protection.json", protection)
    monkeypatch.setattr(module, "verify_product_protection", lambda *_: protection)
    program = {
        "schema_version": "forge.compose_lipid_source_event.v1",
        "policy": {"training_calls": 0, "source_flags_changed": False},
        "config": pin(tmp_path, tmp_path / "program_config.json"),
        "artifacts": {"programs.jsonl.gz": pin(tmp_path, tmp_path / "programs.gz")},
    }
    dump(tmp_path / "program.json", program)
    config = {
        "schema_version": module.CONFIG_SCHEMA,
        "policy": module.POLICY,
        "protection_result": pin(tmp_path, tmp_path / "protection.json"),
        "program_results": [pin(tmp_path, tmp_path / "program.json")],
    }
    dump(tmp_path / "config.json", config)
    for name in module.IMPLEMENTATION:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    return tmp_path


def test_build_replay_verification_and_metadata_only_sqlite_access(workspace):
    result = module.build_precursor_audit(workspace, Path("config.json"), Path("out"))
    assert module.verify_precursor_audit(workspace, Path("out/result.json")) == result
    replay = module.build_precursor_audit(workspace, Path("config.json"), Path("again"))
    assert result["summary"] == replay["summary"]
    assert {key: r["sha256"] for key, r in result["artifacts"].items()} == {
        key: r["sha256"] for key, r in replay["artifacts"].items()
    }
    assert (
        result["summary"]["panels"]["unseen_precursor_identity"][
            "unseen_precursor_label_claim_contradicted"
        ]
        == 1
    )


def test_changed_pinned_program_bytes_fail_before_publishing(workspace):
    (workspace / "programs.gz").write_bytes(b"corrupt")
    with pytest.raises(PinError):
        module.build_precursor_audit(workspace, Path("config.json"), Path("out"))
    assert not (workspace / "out").exists()


@pytest.mark.parametrize("defect", ["summary", "rows", "identity", "scope"])
def test_rehashed_forgery_does_not_pass_independent_recount(workspace, defect):
    result = module.build_precursor_audit(workspace, Path("config.json"), Path("out"))
    if defect == "summary":
        result["summary"]["training_ready"] = True
    elif defect == "scope":
        result["policy"] = {**result["policy"], "training_rows_admitted": 1}
    else:
        name = "rows.jsonl.gz" if defect == "rows" else "identities.jsonl.gz"
        path = workspace / result["artifacts"][name]["path"]
        rows = list(module._read(path))
        if defect == "rows":
            rows[0]["training_admitted"] = True
        else:
            rows[0]["historical_fold"] = "heldout"
        write_rows(path, rows)
        result["artifacts"][name] = pin(workspace, path)
    dump(workspace / "out/result.json", result)
    with pytest.raises(ComposeLipidError):
        module.verify_precursor_audit(workspace, Path("out/result.json"))


def test_failed_serialization_publishes_no_partial_output(workspace, monkeypatch):
    original = module._json

    def broken(value):
        if value.get("training_admitted") is False:
            raise RuntimeError("write failure")
        return original(value)

    monkeypatch.setattr(module, "_json", broken)
    with pytest.raises(RuntimeError, match="write failure"):
        module.build_precursor_audit(workspace, Path("config.json"), Path("out"))
    assert not (workspace / "out").exists()
    assert not list(workspace.glob(".precursors-*"))


def test_existing_output_cannot_be_overwritten(workspace):
    (workspace / "out").mkdir()
    with pytest.raises(ComposeLipidError, match="fresh"):
        module.build_precursor_audit(workspace, Path("config.json"), Path("out"))
