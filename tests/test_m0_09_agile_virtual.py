from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from forge.corpus.agile_virtual import (
    AgileVirtualExtractionError,
    extract_agile_virtual_smiles,
    write_agile_virtual_smiles,
)

REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.csv"
    source.write_text("smiles,descriptor\nCCO,1\nCCN,2\n")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "m0_09_agile_virtual_smiles_config.v1",
                "task": "fixture",
                "generated_utc": "2026-07-29T00:00:00+00:00",
                "source": {
                    "asset": source.name,
                    "doi": "fixture",
                    "expected_sha256": _sha256(source),
                    "expected_bytes": source.stat().st_size,
                    "expected_header_columns": 2,
                    "smiles_column": "smiles",
                },
                "expected_counts": {
                    "source_rows": 2,
                    "valid_smiles": 2,
                    "unique_source_smiles": 2,
                    "unique_canonical_smiles": 2,
                },
                "claims_boundary": {
                    "l2_route_supervision_present": False,
                },
            },
            sort_keys=True,
        )
    )
    return config, source


def test_extracts_only_smiles_with_deterministic_gzip(tmp_path: Path) -> None:
    config, source = _write_fixture(tmp_path)

    first_manifest, first_payload = extract_agile_virtual_smiles(config, source)
    second_manifest, second_payload = extract_agile_virtual_smiles(config, source)

    assert first_manifest == second_manifest
    assert first_payload == second_payload
    decompressed = gzip.decompress(first_payload).decode().splitlines()
    assert decompressed[0] == ("source_row_index,source_smiles,canonical_isomeric_smiles")
    assert decompressed[1:] == ["0,CCO,CCO", "1,CCN,CCN"]
    assert first_manifest["claims_boundary"]["l2_route_supervision_present"] is False


def test_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    config, source = _write_fixture(tmp_path)
    source.write_text(source.read_text() + "\n")

    with pytest.raises(AgileVirtualExtractionError, match="hash mismatch"):
        extract_agile_virtual_smiles(config, source)


def test_writer_is_atomic_and_reproducible(tmp_path: Path) -> None:
    config, source = _write_fixture(tmp_path)
    manifest, payload = extract_agile_virtual_smiles(config, source)
    output = tmp_path / "derived.csv.gz"
    manifest_path = tmp_path / "manifest.json"

    write_agile_virtual_smiles(manifest, payload, output, manifest_path)
    first_output = output.read_bytes()
    first_manifest = manifest_path.read_bytes()
    write_agile_virtual_smiles(manifest, payload, output, manifest_path)

    assert output.read_bytes() == first_output
    assert manifest_path.read_bytes() == first_manifest
    assert not list(tmp_path.glob(".*.tmp"))


def test_committed_derivative_preserves_12276_structure_only_rows() -> None:
    output = REPO / "data/derived/agile_virtual12k_smiles.csv.gz"
    manifest_path = REPO / "data/derived/agile_virtual12k_smiles.manifest.json"
    if not output.exists() or not manifest_path.exists():
        pytest.fail("committed AGILE virtual SMILES derivative is missing")
    manifest = json.loads(manifest_path.read_text())

    assert manifest["summary"]["source_rows"] == 12276
    assert manifest["summary"]["unique_canonical_smiles"] == 12276
    assert manifest["claims_boundary"]["l2_route_supervision_present"] is False
    assert _sha256(output) == manifest["output"]["sha256"]
    with gzip.open(output, "rt", newline="") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 12276
