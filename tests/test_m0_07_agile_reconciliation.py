from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from forge.potency import agile_reconciliation
from forge.potency.agile_reconciliation import (
    AgileReconciliationError,
    _canonical,
    _load_config,
    _normalized_label,
    _publish_atomically,
    _render_csv_gzip,
    _source_matrix,
    _verify_input,
)


def _staging_leftovers(parent: Path, name: str) -> list[Path]:
    return [entry for entry in parent.iterdir() if entry.name.startswith(f".{name}.")]


def test_publish_writes_every_artifact(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _publish_atomically(output, {"a.json": b"1", "b.csv.gz": b"2"})
    assert (output / "a.json").read_bytes() == b"1"
    assert (output / "b.csv.gz").read_bytes() == b"2"
    assert not _staging_leftovers(tmp_path, "run")


def test_publish_replaces_a_previous_run_wholesale(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _publish_atomically(output, {"a.json": b"old", "stale.csv": b"old"})
    _publish_atomically(output, {"a.json": b"new"})
    assert (output / "a.json").read_bytes() == b"new"
    # The rename swaps the directory, so a file the new run does not emit does not survive.
    assert not (output / "stale.csv").exists()
    assert not _staging_leftovers(tmp_path, "run")


def test_failed_publish_leaves_the_previous_run_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publish that dies *while moving files* must not leave a half-swapped directory.

    The defect this pins: artifacts were moved into place one `os.replace` at a time, so a failure
    partway through left new data files beside the previous run's result document, whose sha256
    manifest then described bytes no longer on disk -- a result set that still parses and is
    silently self-inconsistent.

    The failure has to be injected into the move itself. Failing earlier, while payloads are still
    being written to staging, cannot tear anything and so does not exercise the defect at all.
    """
    output = tmp_path / "run"
    _publish_atomically(output, {"result.json": b"v1", "data.csv.gz": b"v1"})

    real_replace = os.replace
    calls = {"n": 0}

    def failing_replace(src: object, dst: object) -> None:
        # Call 1 moves the old output aside; call 2 is the publish itself and is the one to break.
        # Call 3 is the restore, which must be allowed to succeed.
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError(28, "No space left on device")
        real_replace(src, dst)

    monkeypatch.setattr(agile_reconciliation.os, "replace", failing_replace)

    with pytest.raises(OSError, match="No space left on device"):
        _publish_atomically(output, {"result.json": b"v2", "data.csv.gz": b"v2"})

    monkeypatch.undo()
    assert (output / "result.json").read_bytes() == b"v1"
    assert (output / "data.csv.gz").read_bytes() == b"v1"
    assert sorted(p.name for p in output.iterdir()) == ["data.csv.gz", "result.json"]
    assert not _staging_leftovers(tmp_path, "run")


def test_previous_run_survives_even_when_the_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the publish fails *and* the restore fails, the old run must still exist somewhere.

    The superseded copy is therefore deleted only after a successful publish. Deleting it in a
    `finally` would destroy the previous run precisely when the restore could not put it back --
    turning a failed publish into data loss.
    """
    output = tmp_path / "run"
    _publish_atomically(output, {"result.json": b"v1"})

    real_replace = os.replace
    calls = {"n": 0}

    def failing_replace(src: object, dst: object) -> None:
        calls["n"] += 1
        if calls["n"] >= 2:  # break the publish and the restore that follows it
            raise OSError(28, "No space left on device")
        real_replace(src, dst)

    monkeypatch.setattr(agile_reconciliation.os, "replace", failing_replace)

    with pytest.raises(OSError, match="No space left on device"):
        _publish_atomically(output, {"result.json": b"v2"})

    monkeypatch.undo()
    preserved = [p for p in tmp_path.rglob("result.json") if p.read_bytes() == b"v1"]
    assert preserved, "the previous run was destroyed by a failed restore"


def test_cleanup_does_not_mask_the_original_error(tmp_path: Path) -> None:
    """The caller must see why the publish failed, not a complaint about the staging directory.

    The previous implementation cleaned up in `finally: staging.rmdir()`. On any failure the
    staging directory still held files, so `rmdir` raised `Directory not empty` and that replaced
    the real exception on its way out.
    """
    output = tmp_path / "run"
    with pytest.raises(TypeError) as caught:
        _publish_atomically(output, {"a.json": b"ok", "b.json": None})  # type: ignore[dict-item]

    assert not isinstance(caught.value, OSError)
    assert "Directory not empty" not in str(caught.value)
    assert not output.exists()
    assert not _staging_leftovers(tmp_path, "run")


REPO = Path(__file__).resolve().parents[1]


def test_component_labels_normalize_both_orders() -> None:
    assert _normalized_label("A17B5C3") == "A17B5C3"
    assert _normalized_label("A17C3B5") == "A17B5C3"


def test_model_canonicalization_intentionally_removes_alkene_stereo() -> None:
    trans = "CCCCCCC/C=C/C(=O)OCCCCCC=O"
    cis = "CCCCCCC/C=C\\C(=O)OCCCCCC=O"

    assert _canonical(trans, isomeric=True, label="trans") != _canonical(
        cis, isomeric=True, label="cis"
    )
    assert _canonical(trans, isomeric=False, label="trans") == _canonical(
        cis, isomeric=False, label="cis"
    )


def test_gzip_rendering_is_byte_deterministic() -> None:
    rows = [{"a": "1", "b": "2"}]
    first = _render_csv_gzip(rows, ("a", "b"))
    second = _render_csv_gzip(rows, ("a", "b"))

    assert first == second
    assert gzip.decompress(first) == b"a,b\n1,2\n"


def test_hash_mismatch_fails_loudly(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x\n")

    with pytest.raises(AgileReconciliationError, match="hash mismatch"):
        _verify_input(source, "0" * 64, "source")


def test_config_rejects_single_graph_mixture_training(tmp_path: Path) -> None:
    config = json.loads((REPO / "configs/bio/m0_07_agile_label_reconciliation.json").read_text())
    config["policy"]["mixture_handling"] = "train_as_single_graph"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    with pytest.raises(AgileReconciliationError, match="mixture_handling"):
        _load_config(path)


@pytest.mark.needs_vendor
def test_official_source_workbook_contains_both_1200_label_matrices() -> None:
    workbook = REPO / "data/vendor/agile_article_source_data.xlsx"
    if not workbook.exists():
        pytest.skip("AGILE official source workbook is not vendored")

    hela = _source_matrix(
        workbook,
        sheet_name="Figure 2",
        header_row=3,
        first_data_row=4,
        first_head_column=2,
    )
    raw = _source_matrix(
        workbook,
        sheet_name="Supplementary Figure 21",
        header_row=2,
        first_data_row=3,
        first_head_column=3,
    )

    assert len(hela) == len(raw) == 1200
    assert hela["A1B1C1"] == pytest.approx(4.05668942)
    assert raw["A1B1C1"] == pytest.approx(1.54475317)


@pytest.mark.needs_vendor
def test_frozen_reconciliation_contract_when_present() -> None:
    result_path = REPO / "results/m0_07/agile_label_reconciliation.json"
    if not result_path.exists():
        pytest.skip("M0-07 AGILE reconciliation result has not been generated")
    result = json.loads(result_path.read_text())

    assert result["schema_version"] == "m0_07_agile_reconciliation.v1"
    assert result["status"] == "completed_blocking_source_reconciliation"
    assert result["summary"]["nominal_experimental_measurements"] == 1200
    assert result["summary"]["curated_single_structure_records"] == 1100
    assert result["summary"]["excluded_mixture_measurements"] == 100
    assert result["summary"]["B5_pure_trans_graph_corrections"] == 100
    assert result["summary"]["unique_model_graphs"] == 1100
    assert all(result["adversarial_checks"].values())
    assert result["downstream_contract"]["raw_1200_row_file_allowed_for_oracle_fit"] is False

    for filename, metadata in result["artifacts"].items():
        path = REPO / "results/m0_07" / filename
        payload = path.read_bytes()
        assert len(payload) == metadata["bytes"]
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]

    with gzip.open(REPO / "results/m0_07/agile_oracle_curated.csv.gz", "rt") as handle:
        curated = list(csv.DictReader(handle))
    with gzip.open(REPO / "results/m0_07/agile_mixture_exclusions.csv.gz", "rt") as handle:
        excluded = list(csv.DictReader(handle))
    with gzip.open(REPO / "results/m0_07/agile_reconciliation_ledger.csv.gz", "rt") as handle:
        ledger = list(csv.DictReader(handle))

    assert len(curated) == 1100
    assert len(excluded) == 100
    assert len(ledger) == 1200
    assert not any("B4" in row["label"] for row in curated)
    assert all("B4" in row["label"] for row in excluded)
    assert all(row["structure_policy"] for row in curated)
    assert len({row["model_smiles"] for row in curated}) == 1100

    b5 = next(row for row in curated if row["label"] == "A1B5C1")
    assert "/" in b5["isomeric_smiles"]
    assert "\\" not in b5["isomeric_smiles"]

    raw = gzip.decompress(
        (REPO / "results/m0_07/agile_oracle_curated.csv.gz").read_bytes()
    ).decode()
    assert len(list(csv.DictReader(io.StringIO(raw)))) == 1100
