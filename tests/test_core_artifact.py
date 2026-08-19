"""Tests for forge.core.artifact.

The properties that matter here are the fail-closed ones. A receipt that records an input it did
not actually verify, or a stage that leaves a partial result after an exception, is worse than no
receipt at all -- it looks like provenance while providing none.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from forge.core.artifact import ArtifactError, artifact_run
from forge.core.hashing import sha256_bytes, sha256_file
from forge.core.io import atomic_write

SCHEMA = "phase1_example_result.v1"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature repository: one pinned input, one output directory."""
    (tmp_path / "data").mkdir()
    source = tmp_path / "data" / "products.csv"
    atomic_write(source, b"id,smiles\np1,CCO\np2,CCN\n")
    (tmp_path / "out").mkdir()
    return tmp_path


def pins(repo: Path) -> dict[str, dict[str, str]]:
    target = repo / "data" / "products.csv"
    return {"products": {"path": "data/products.csv", "sha256": str(sha256_file(target))}}


# ------------------------------------------------------------------ the happy path


def test_writes_outputs_and_a_receipt(repo: Path) -> None:
    with artifact_run(SCHEMA, repo, repo / "out", inputs=pins(repo), task="example") as run:
        rows = run.read_csv("products")
        run.write_csv("selected.csv.gz", rows, ["id", "smiles"])
        run.summary = {"selected": len(rows)}

    receipt = json.loads((repo / "out" / "result.json").read_text())
    assert receipt["schema_version"] == SCHEMA
    assert receipt["status"] == "complete"
    assert receipt["task"] == "example"
    assert receipt["summary"] == {"selected": 2}
    assert receipt["inputs"]["products"]["path"] == "data/products.csv"
    assert receipt["artifacts"]["selected.csv.gz"]["columns"] == ["id", "smiles"]


def test_output_hash_in_the_receipt_matches_the_file_written(repo: Path) -> None:
    """The receipt's claim about an output has to be true of the bytes on disk."""
    with artifact_run(SCHEMA, repo, repo / "out", inputs=pins(repo)) as run:
        run.write_csv("selected.csv.gz", run.read_csv("products"), ["id", "smiles"])

    receipt = json.loads((repo / "out" / "result.json").read_text())
    written = repo / "out" / "selected.csv.gz"
    assert receipt["artifacts"]["selected.csv.gz"]["sha256"] == sha256_file(written)
    assert receipt["artifacts"]["selected.csv.gz"]["bytes"] == written.stat().st_size


def test_receipts_are_reproducible(repo: Path) -> None:
    """Two identical runs must produce identical bytes, or downstream pins break on every rerun."""
    produced = []
    for name in ("a", "b"):
        out = repo / name
        out.mkdir()
        with artifact_run(SCHEMA, repo, out, inputs=pins(repo)) as run:
            run.write_csv("selected.csv.gz", run.read_csv("products"), ["id", "smiles"])
            run.summary = {"selected": 2}
        produced.append((out / "result.json").read_bytes())
    assert produced[0] == produced[1]


def test_timestamps_are_opt_in(repo: Path) -> None:
    """A stamped receipt hashes differently every run, so it must never be the default."""
    with artifact_run(SCHEMA, repo, repo / "out", inputs=pins(repo)) as run:
        run.summary = {"n": 1}
    assert "generated_utc" not in json.loads((repo / "out" / "result.json").read_text())

    stamped = repo / "stamped"
    stamped.mkdir()
    with artifact_run(SCHEMA, repo, stamped, inputs=pins(repo), record_time=True) as run:
        run.summary = {"n": 1}
    assert "generated_utc" in json.loads((stamped / "result.json").read_text())


def test_seed_is_recorded(repo: Path) -> None:
    with artifact_run(SCHEMA, repo, repo / "out", inputs=pins(repo), seed=20260818) as run:
        run.summary = {"n": 1}
    receipt = json.loads((repo / "out" / "result.json").read_text())
    assert receipt["randomness"] == {"seed": 20260818}


# ------------------------------------------------------------------ failing closed


def test_a_changed_input_is_rejected(repo: Path) -> None:
    declared = pins(repo)
    atomic_write(repo / "data" / "products.csv", b"id,smiles\np1,CHANGED\n")
    with pytest.raises(ArtifactError, match="changed"):
        with artifact_run(SCHEMA, repo, repo / "out", inputs=declared) as run:
            run.read_csv("products")


def test_an_undeclared_input_is_rejected(repo: Path) -> None:
    """A stage may only read what its config declared, or the receipt would be incomplete."""
    with pytest.raises(ArtifactError, match="undeclared input"):
        with artifact_run(SCHEMA, repo, repo / "out", inputs=pins(repo)) as run:
            run.read_csv("something_else")


def test_nothing_is_written_when_the_stage_raises(repo: Path) -> None:
    """The property that matters most: a failed stage leaves no partial result."""
    with pytest.raises(ZeroDivisionError):
        with artifact_run(SCHEMA, repo, repo / "out", inputs=pins(repo)) as run:
            run.write_csv("selected.csv.gz", run.read_csv("products"), ["id", "smiles"])
            run.summary = {"selected": 1 / 0}

    assert list((repo / "out").iterdir()) == []


def test_a_previous_result_survives_a_failed_rerun(repo: Path) -> None:
    """Re-running a stage that fails must not destroy the last good result."""
    with artifact_run(SCHEMA, repo, repo / "out", inputs=pins(repo)) as run:
        run.summary = {"selected": 2}
    original = (repo / "out" / "result.json").read_bytes()

    with pytest.raises(RuntimeError):
        with artifact_run(SCHEMA, repo, repo / "out", inputs=pins(repo)) as run:
            run.summary = {"selected": 999}
            raise RuntimeError("stage failed")

    assert (repo / "out" / "result.json").read_bytes() == original


def test_writing_the_same_output_twice_is_rejected(repo: Path) -> None:
    with pytest.raises(ArtifactError, match="written twice"):
        with artifact_run(SCHEMA, repo, repo / "out", inputs=pins(repo)) as run:
            run.write_json("x.json", {"a": 1})
            run.write_json("x.json", {"a": 2})


def test_only_inputs_actually_read_are_recorded(repo: Path) -> None:
    """The receipt describes what the stage read, not what it was offered."""
    declared = pins(repo) | {"unused": {"path": "data/products.csv", "sha256": "0" * 64}}
    with artifact_run(SCHEMA, repo, repo / "out", inputs=declared) as run:
        run.read_csv("products")
    receipt = json.loads((repo / "out" / "result.json").read_text())
    assert set(receipt["inputs"]) == {"products"}


def test_inputs_are_verified_once(repo: Path) -> None:
    """Repeated reads reuse the verified path rather than re-hashing a large ledger."""
    with artifact_run(SCHEMA, repo, repo / "out", inputs=pins(repo)) as run:
        assert run.input("products") == run.input("products")


# ------------------------------------------------------------------ round trip


def test_receipt_survives_the_pin_verifier(repo: Path) -> None:
    """A receipt this writes must be checkable by the same rules verify_artifact_pins applies."""
    with artifact_run(SCHEMA, repo, repo / "out", inputs=pins(repo)) as run:
        run.write_csv("selected.csv.gz", run.read_csv("products"), ["id", "smiles"])

    receipt = json.loads((repo / "out" / "result.json").read_text())
    record = receipt["inputs"]["products"]
    assert sha256_file(repo / record["path"]) == record["sha256"]

    body = gzip.decompress((repo / "out" / "selected.csv.gz").read_bytes())
    assert body.startswith(b"id,smiles\n")
    assert (
        sha256_bytes((repo / "out" / "selected.csv.gz").read_bytes())
        == (receipt["artifacts"]["selected.csv.gz"]["sha256"])
    )
