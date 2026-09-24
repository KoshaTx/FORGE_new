"""Full-universe accounting preserves every row and fails closed before training."""

import gzip
import json
import shutil
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import pytest

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import sha256_file
from forge.corpus import compose_lipid_universe as module

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value))


def pin(repo, path):
    return {"path": path.relative_to(repo).as_posix(), "sha256": str(sha256_file(path))}


def audit_row(target, family, exact, reasons):
    return {
        "target_id": target,
        "family": family,
        "exact_program_evidence": exact,
        "clear_of_known_exclusions": not reasons,
        "known_exclusion_reasons": reasons,
        "training_admitted": False,
    }


def write_rows(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    rows = [
        ("blocked", "a", "train", "CCN", 20),
        ("clear", "a", "train", "CCC", 78),
        ("no_program", "b", "train", "CCO", 100),
        ("heldout", "a", "heldout", "INVALID_UNPARSED", 60),
        ("calibration", "b", "calibration", "CAL_UNPARSED", 50),
        ("quarantine", "b", "quarantine", "Q_UNPARSED", 70),
        ("reference", "ref", "reference", "REF_UNPARSED", 35),
        ("heldout_alias", "b", None, "INVALID_UNPARSED", 60),
        ("blocked_alias", "b", None, "CCN", 20),
        ("large", "a", None, "UNASSIGNED_UNPARSED", 254),
        ("duplicate", "b", None, "CCC", 78),
    ]
    with sqlite3.connect(tmp_path / "corpus.sqlite") as db:
        db.execute(
            "CREATE TABLE targets(target_id TEXT PRIMARY KEY, family TEXT, constitution TEXT, source_anchor INTEGER, heavy_atoms INTEGER, size_disposition TEXT)"
        )
        db.execute(
            "CREATE TABLE assignments(target_id TEXT PRIMARY KEY, family TEXT, provider_split TEXT, forge_split TEXT)"
        )
        db.execute(
            "CREATE TABLE duplicate_constitutions(constitution TEXT PRIMARY KEY, source_rows INTEGER)"
        )
        for target, family, split, string, atoms in rows:
            db.execute(
                "INSERT INTO targets VALUES (?,?,?,?,?,?)",
                (
                    target,
                    family,
                    string,
                    0,
                    atoms,
                    "above_80_heavy_atom_hold" if atoms > 80 else "model_supported",
                ),
            )
            if split:
                db.execute(
                    "INSERT INTO assignments VALUES (?,?,?,?)", (target, family, split, split)
                )
        db.execute(
            "INSERT INTO duplicate_constitutions SELECT constitution,count(*) FROM targets GROUP BY constitution HAVING count(*)>1"
        )
    audit_rows = [
        audit_row("blocked", "a", True, ["historical_protected_precursor"]),
        audit_row("clear", "a", True, []),
        audit_row("no_program", "b", False, ["no_exact_program_evidence"]),
    ]
    write_rows(tmp_path / "audit.gz", audit_rows)
    dump(tmp_path / "catalogue.json", {"a": {}, "b": {}})
    families = dict(Counter(row[1] for row in rows))
    imported = {
        "artifacts": {
            "corpus.sqlite": pin(tmp_path, tmp_path / "corpus.sqlite"),
            "program_catalogue.json": pin(tmp_path, tmp_path / "catalogue.json"),
        },
        "summary": {"universe_by_family": families, "universe_rows": len(rows)},
    }
    dump(tmp_path / "import.json", imported)
    dump(
        tmp_path / "protection.json",
        {"inputs": {"import_result": pin(tmp_path, tmp_path / "import.json")}},
    )
    audited = {
        "inputs": {"protection_result": pin(tmp_path, tmp_path / "protection.json")},
        "artifacts": {"rows.jsonl.gz": pin(tmp_path, tmp_path / "audit.gz")},
    }
    dump(tmp_path / "audit.json", audited)
    # Imported receipts have their own replay suites. This fixture isolates their consumer.
    monkeypatch.setattr(module, "verify_compose_lipid", lambda *_: imported)
    monkeypatch.setattr(module, "verify_precursor_audit", lambda *_: audited)
    for name in module.IMPLEMENTATION:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    config = {
        "schema_version": module.CONFIG_SCHEMA,
        "policy": module.POLICY,
        "expected_family_rows": dict(families),
        "reference_families": ["ref"],
        "inputs": {
            "import_result": pin(tmp_path, tmp_path / "import.json"),
            "precursor_audit_result": pin(tmp_path, tmp_path / "audit.json"),
        },
    }
    dump(tmp_path / "config.json", config)
    return tmp_path, config, audit_rows


def test_every_record_retained_without_product_parsing_and_source_unchanged(workspace):
    repo, _, _ = workspace
    before = sha256_file(repo / "corpus.sqlite")
    result = module.build_universe_readiness(repo, Path("config.json"), Path("output"))
    rows = {row["target_id"]: row for row in module._read(repo / "output/readiness.jsonl.gz")}
    assert result["summary"]["totals"]["rows"] == len(rows) == 11
    assert result["summary"]["program_families"] == 2
    assert result["summary"]["totals"]["outside_200k_selection"] == 4
    assert result["summary"]["totals"]["existing_exact_program_evidence"] == 2
    assert rows["large"]["source_declared_heavy_atoms"] == 254
    assert rows["large"]["disposition"] == "pending_partition_before_decomposition"
    assert rows["duplicate"]["source_string_multiplicity"] == 2
    assert rows["duplicate"]["disposition"] == "pending_partition_before_decomposition"
    assert rows["clear"]["disposition"] == "pending_global_qualification"
    for name in ("heldout_alias", "blocked_alias"):
        assert rows[name]["disposition"] == "excluded_exact_source_alias_of_known_exclusion"
    assert all(row["training_admitted"] is False for row in rows.values())
    assert "INVALID_UNPARSED" not in json.dumps(rows)
    assert sha256_file(repo / "corpus.sqlite") == before
    assert module.verify_universe_readiness(repo, Path("output/result.json")) == result


def test_replay_and_compressed_ledger_are_deterministic(workspace):
    repo, _, _ = workspace
    first = module.build_universe_readiness(repo, Path("config.json"), Path("one"))
    second = module.build_universe_readiness(repo, Path("config.json"), Path("two"))
    assert first["summary"] == second["summary"]
    assert (
        first["artifacts"]["readiness.jsonl.gz"]["sha256"]
        == second["artifacts"]["readiness.jsonl.gz"]["sha256"]
    )
    with pytest.raises(ComposeLipidError, match="fresh"):
        module.build_universe_readiness(repo, Path("config.json"), Path("one"))


@pytest.mark.parametrize(
    "defect", ["admitted", "remove", "append", "promote_unassigned", "integer_boolean"]
)
def test_rehashed_ledger_tampering_is_rejected(workspace, defect):
    repo, _, _ = workspace
    result = module.build_universe_readiness(repo, Path("config.json"), Path("output"))
    ledger = repo / "output/readiness.jsonl.gz"
    rows = list(module._read(ledger))
    if defect == "admitted":
        rows[0]["training_admitted"] = True
    elif defect == "integer_boolean":
        rows[0]["training_admitted"] = 0
    elif defect == "remove":
        rows.pop()
    elif defect == "append":
        rows.append(rows[-1])
    else:
        next(row for row in rows if row["target_id"] == "large")["forge_split"] = "train"
    write_rows(ledger, rows)
    result["artifacts"][ledger.name] = pin(repo, ledger)
    dump(repo / "output/result.json", result)
    with pytest.raises(ComposeLipidError, match="ledger does not reproduce"):
        module.verify_universe_readiness(repo, Path("output/result.json"))


@pytest.mark.parametrize("defect", ["summary", "policy", "implementation"])
def test_receipt_cannot_claim_training_readiness(workspace, defect):
    repo, _, _ = workspace
    result = module.build_universe_readiness(repo, Path("config.json"), Path("output"))
    if defect == "summary":
        result["summary"]["training_ready"] = True
    elif defect == "policy":
        result["policy"] = {**result["policy"], "record_cap": 200000}
    else:
        result["implementation"].clear()
    dump(repo / "output/result.json", result)
    with pytest.raises(ComposeLipidError):
        module.verify_universe_readiness(repo, Path("output/result.json"))


@pytest.mark.parametrize(
    "defect",
    [
        "missing_train",
        "extra_target",
        "family",
        "clear_without_program",
        "admitted",
        "unknown_reason",
        "duplicate",
    ],
)
def test_incomplete_or_inconsistent_audit_never_silently_joins(workspace, defect):
    repo, _, rows = workspace
    if defect == "missing_train":
        rows.pop()
    elif defect == "extra_target":
        rows.append(audit_row("large", "a", True, []))
    elif defect == "family":
        rows[0]["family"] = "b"
    elif defect == "clear_without_program":
        rows[1]["exact_program_evidence"] = False
    elif defect == "admitted":
        rows[0]["training_admitted"] = True
    elif defect == "unknown_reason":
        rows[0]["known_exclusion_reasons"].append("unrecognized")
    else:
        rows.append(rows[0])
    with closing(sqlite3.connect((repo / "corpus.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        with pytest.raises((ComposeLipidError, sqlite3.IntegrityError)):
            list(module.universe_rows(db, rows))


@pytest.mark.parametrize("defect", ["split", "family", "orphan"])
def test_source_assignment_errors_are_not_dropped(workspace, defect):
    repo, _, rows = workspace
    with sqlite3.connect(repo / "corpus.sqlite") as db:
        if defect == "split":
            db.execute("UPDATE assignments SET forge_split='mystery' WHERE target_id='heldout'")
        elif defect == "family":
            db.execute("UPDATE assignments SET family='b' WHERE target_id='heldout'")
        else:
            db.execute("INSERT INTO assignments VALUES ('missing','a','heldout','heldout')")
        with pytest.raises(ComposeLipidError, match="assignment"):
            list(module.universe_rows(db, rows))


def test_failed_census_publishes_no_partial_output(workspace):
    repo, config, _ = workspace
    config["expected_family_rows"]["a"] -= 1
    dump(repo / "config.json", config)
    with pytest.raises(ComposeLipidError, match="counts"):
        module.build_universe_readiness(repo, Path("config.json"), Path("output"))
    assert not (repo / "output").exists()
