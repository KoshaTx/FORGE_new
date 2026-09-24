"""Complete-family weighting, constitutional deduplication and closed admission boundaries."""

import json
import math
import shutil
import sqlite3
from collections import Counter

import pytest

from forge.core.hashing import PinError, resolve_pin
from forge.corpus import compose_lipid_training_measure as measure
from forge.corpus.compose_lipid_source_view import dump, pin


def inputs(repo, *, missing=False, pending=0, duplicate=None, reference=False, mismatch=False):
    """Small synthetic metadata fixture with all 23 formal families, not chemical data."""
    implementation = repo / "forge/corpus/compose_lipid_training_measure.py"
    implementation.parent.mkdir(parents=True)
    shutil.copyfile(measure.__file__, implementation)
    producer = repo / "qualification.py"
    producer.write_text("# Synthetic metadata fixture producer.\n")
    corpus = repo / "source.bin"
    corpus.write_bytes(b"synthetic source fixture\n")

    def save(name, document):
        path = repo / name
        dump(path, document)
        return pin(repo, path)

    families = [f"family_{i:02}" for i in range(23)]
    expected = {f: 4 for f in families} | {"reference": 1}
    imported = save(
        "import.json",
        {
            "schema_version": "forge.compose_lipid_import.v1",
            "sampling_policy": "equal_family_mass_after_qualification",
            "summary": {"universe_by_family": expected, "universe_rows": sum(expected.values())},
            "artifacts": {"corpus.sqlite": pin(repo, corpus)},
        },
    )
    universe = save(
        "universe.json",
        {
            "schema_version": "forge.compose_lipid_universe_config.v1",
            "expected_family_rows": expected,
            "reference_families": ["reference"],
            "inputs": {"import_result": imported},
            "policy": {"intended_training_measure": "equal_family_mass_after_all_admission_gates"},
        },
    )
    records = []
    for i, family in enumerate(families[:-1] if missing else families):
        for j in range(3 if i == 0 else 1):
            records.append((f"target-{i}-{j}", f"graph-{i}-{j}", family, f"binding-{j}", i, j))
    if reference:
        records.append(("reference", "reference-graph", "reference", "reference", 23, 0))
    if duplicate:
        row = list(records[1])
        row[0 if duplicate == "target" else 1] = records[0][0 if duplicate == "target" else 1]
        records[1] = tuple(row)
    source = repo / "preparation.sqlite"
    with sqlite3.connect(source) as db:
        db.execute(
            "CREATE TABLE records(target_id TEXT,constitution_id TEXT,family TEXT,"
            "program_key TEXT,shard_id INTEGER,row_index INTEGER)"
        )
        # Physical insertion order differs from model lookup order.
        db.executemany("INSERT INTO records VALUES (?,?,?,?,?,?)", list(reversed(records)))
    evidence_db = repo / "evidence.sqlite"
    with sqlite3.connect(evidence_db) as db:
        db.execute("CREATE TABLE exact(target_id TEXT,constitution_id TEXT,family TEXT)")
        db.executemany("INSERT INTO exact VALUES (?,?,?)", [r[:3] for r in records])
        if mismatch:
            db.execute(
                "UPDATE exact SET constitution_id='different' WHERE target_id=?", (records[0][0],)
            )
    evidence_pin = save(
        "evidence.json",
        {
            "schema_version": "forge.compose_lipid_evidence_index.v1",
            "training_admitted": False,
            "artifact": pin(repo, evidence_db),
            "partition_inputs": {"corpus": pin(repo, corpus)},
            "summary": {
                "eligible": len(records) + pending,
                "exact": len(records),
                "pending": pending,
            },
        },
    )
    request = save(
        "request.json",
        {
            "implementation": pin(repo, producer),
            "inputs": {"evidence_index": evidence_pin, "evidence_database": pin(repo, evidence_db)},
        },
    )
    totals = {"records": len(records), "atoms": 200, "above_96_atoms": 1}
    population = save(
        "population.json",
        {
            "schema_version": "forge.unified_preparation_index.v1",
            "training_admitted": False,
            "sampling_weights_fitted": False,
            "request": request,
            "artifacts": {"preparation.sqlite": pin(repo, source)},
            "by_family": dict(Counter(r[2] for r in records)),
            "totals": totals,
            "tensor_shards": 23,
        },
    )
    verified = save(
        "verification.json",
        {
            "schema_version": "forge.unified_preparation_verification.v1",
            "implementation": pin(repo, producer),
            "inputs": {"population": population},
            "totals": totals,
            "verified_tensor_shards": 23,
            "all_exact_eligible_records_present_once": True,
            "source_identity_family_size_and_constitution_preserved": True,
            "shard_row_indices_complete": True,
            "vocabulary_remaps_lossless": True,
        },
    )
    return {
        "universe_config": universe,
        "population": population,
        "verification": verified,
        "policy": measure.POLICY,
    }, records


def test_equal_family_mass_preserves_every_graph_and_ignores_alias_counts(tmp_path):
    kwargs, records = inputs(tmp_path)
    result = measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)
    document = json.loads(result.read_text())
    database = resolve_pin(document["artifact"], tmp_path, label="test measure")
    with sqlite3.connect(database) as db:
        actual = db.execute(
            "SELECT target_id,constitution_id,family,probability FROM weights ORDER BY record_index"
        ).fetchall()
    assert [r[:3] for r in actual] == [r[:3] for r in records]
    # The first family has three rows and three source namespaces; each other family has one.
    assert len(actual) == 25
    assert [r[3] for r in actual[:3]] == [1 / 69] * 3
    assert [r[3] for r in actual[3:]] == [1 / 23] * 22
    assert math.fsum(r[3] for r in actual) == pytest.approx(1, abs=1e-15)
    for family in document["formal_families"]:
        assert math.fsum(r[3] for r in actual if r[2] == family) == pytest.approx(1 / 23)
    assert document["training_admitted"] is False and document["training_calls"] == 0
    assert document["source_role_bindings_receive_separate_mass"] is False


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"missing": True}, "family_22"),
        ({"pending": 7}, "7 eligible records lack exact chemistry"),
        ({"missing": True, "pending": 5}, "5 eligible records lack exact chemistry"),
    ],
)
def test_no_weights_or_partial_output_when_full_qualification_is_absent(tmp_path, options, reason):
    kwargs, _ = inputs(tmp_path, **options)
    with pytest.raises(measure.TrainingMeasureUnavailableError, match=reason) as caught:
        measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)
    assert caught.value.blockers
    assert not (tmp_path / "weighted").exists()
    assert not list(tmp_path.glob(".training-measure-*"))


@pytest.mark.parametrize("kind", ["target", "constitution"])
def test_duplicate_provenance_never_receives_extra_probability(tmp_path, kind):
    kwargs, _ = inputs(tmp_path, duplicate=kind)
    with pytest.raises(ValueError, match="duplicates"):
        measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)
    assert not (tmp_path / "weighted").exists()


def test_reference_rows_cannot_supply_missing_formal_family_mass(tmp_path):
    kwargs, _ = inputs(tmp_path, missing=True, reference=True)
    with pytest.raises(ValueError, match="Nonformal"):
        measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)


def test_misjoined_exact_evidence_is_rejected(tmp_path):
    kwargs, _ = inputs(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="qualified exact source identities"):
        measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)


def test_missing_independent_verification_is_rejected(tmp_path):
    kwargs, _ = inputs(tmp_path)
    path = tmp_path / "verification.json"
    doc = json.loads(path.read_text())
    doc["vocabulary_remaps_lossless"] = False
    dump(path, doc)
    kwargs["verification"] = pin(tmp_path, path)
    with pytest.raises(ValueError, match="independent"):
        measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)


def test_changed_source_lookup_checksum_is_rejected(tmp_path):
    kwargs, _ = inputs(tmp_path)
    with sqlite3.connect(tmp_path / "preparation.sqlite") as db:
        db.execute("DELETE FROM records WHERE row_index=2")
    with pytest.raises(PinError):
        measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)


def test_explicit_v8_policy_required_and_complete_output_immutable(tmp_path):
    kwargs, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="Unsupported"):
        measure.compile_training_measure(
            tmp_path, tmp_path / "weighted", **{**kwargs, "policy": "raw_counts"}
        )
    result = measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)
    before = result.read_bytes()
    with pytest.raises(FileExistsError):
        measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)
    assert result.read_bytes() == before


def test_repeat_build_is_byte_identical_for_saved_probability_database(tmp_path):
    kwargs, _ = inputs(tmp_path)
    first = measure.compile_training_measure(tmp_path, tmp_path / "one", **kwargs)
    second = measure.compile_training_measure(tmp_path, tmp_path / "two", **kwargs)
    a, b = (json.loads(p.read_text()) for p in (first, second))
    assert a["artifact"]["sha256"] == b["artifact"]["sha256"]
    assert a["by_family"] == b["by_family"]


@pytest.mark.parametrize("change", ["family_count", "family_policy", "source_release"])
def test_all_family_v8_contract_cannot_be_replaced_by_a_subset_or_r1(tmp_path, change):
    kwargs, _ = inputs(tmp_path)
    path = tmp_path / "universe.json"
    document = json.loads(path.read_text())
    if change == "family_count":
        del document["expected_family_rows"]["family_22"]
    elif change == "family_policy":
        document["policy"]["intended_training_measure"] = "equal_program_binding_mass"
    else:
        imported = json.loads((tmp_path / "import.json").read_text())
        imported["schema_version"] = "forge.r1"
        dump(tmp_path / "import.json", imported)
        document["inputs"]["import_result"] = pin(tmp_path, tmp_path / "import.json")
    dump(path, document)
    kwargs["universe_config"] = pin(tmp_path, path)
    with pytest.raises(ValueError):
        measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)
    assert not (tmp_path / "weighted").exists()


def test_failure_after_compilation_publishes_no_partial_artifact(tmp_path, monkeypatch):
    kwargs, _ = inputs(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("receipt write failure")

    monkeypatch.setattr(measure, "dump", fail)
    with pytest.raises(OSError, match="receipt write failure"):
        measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)
    assert not (tmp_path / "weighted").exists()
    assert not list(tmp_path.glob(".training-measure-*"))


def add_qualified_cohort(repo, kwargs):
    population = json.loads((repo / kwargs["population"]["path"]).read_text())
    request = json.loads((repo / population["request"]["path"]).read_text())
    evidence_pin = request["inputs"]["evidence_index"]
    evidence = json.loads((repo / evidence_pin["path"]).read_text())
    universe = json.loads((repo / kwargs["universe_config"]["path"]).read_text())
    formal = set(universe["expected_family_rows"]) - set(universe["reference_families"])
    path = repo / "cohort.json"
    dump(
        path,
        {
            "schema_version": "forge.compose_lipid_qualified_cohort.v1",
            "selection": "all_exact_eligible_records_in_pinned_population",
            "policy": measure.COHORT_POLICY,
            "holdouts": "unchanged",
            "authorization": {
                "qualified_cohort_training": True,
                "user_instruction": "Synthetic explicit cohort authorization",
            },
            "inputs": {
                "population": kwargs["population"],
                "verification": kwargs["verification"],
                "evidence_index": evidence_pin,
            },
            "families": sorted(population["by_family"]),
            "by_family": population["by_family"],
            "records": population["totals"]["records"],
            "pending_records_excluded": evidence["summary"]["pending"],
            "formal_families_without_training_support": sorted(
                formal - set(population["by_family"])
            ),
        },
    )
    return {**kwargs, "policy": measure.COHORT_POLICY, "cohort": pin(repo, path)}


def test_explicit_qualified_cohort_preserves_exact_rows_and_exclusions(tmp_path):
    kwargs, records = inputs(tmp_path, missing=True, pending=7)
    kwargs = add_qualified_cohort(tmp_path, kwargs)
    path = measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)
    result = json.loads(path.read_text())
    assert result["records"] == len(records)
    assert result["pending_records_excluded"] == 7
    assert len(result["formal_families"]) == 22
    assert len(result["source_formal_families"]) == 23
    assert all(v["total_probability"] == 1 / 22 for v in result["by_family"].values())
    assert result["inputs"]["cohort"] == kwargs["cohort"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("authorization", {"qualified_cohort_training": False}),
        ("records", 1),
        ("pending_records_excluded", 0),
        ("formal_families_without_training_support", []),
        ("holdouts", "reassigned"),
        ("selection", "target_selected"),
    ],
)
def test_cohort_cannot_hide_missing_evidence_or_change_scope(tmp_path, field, value):
    kwargs, _ = inputs(tmp_path, missing=True, pending=7)
    kwargs = add_qualified_cohort(tmp_path, kwargs)
    path = tmp_path / "cohort.json"
    cohort = json.loads(path.read_text())
    cohort[field] = value
    dump(path, cohort)
    kwargs["cohort"] = pin(tmp_path, path)
    with pytest.raises(ValueError, match="cohort"):
        measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)
    assert not (tmp_path / "weighted").exists()


def test_scoped_measure_still_rejects_nonexact_graph_identity(tmp_path):
    kwargs, _ = inputs(tmp_path, missing=True, pending=7, mismatch=True)
    kwargs = add_qualified_cohort(tmp_path, kwargs)
    with pytest.raises(ValueError, match="exact source identities"):
        measure.compile_training_measure(tmp_path, tmp_path / "weighted", **kwargs)
