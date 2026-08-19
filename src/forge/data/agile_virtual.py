"""Deterministically extract the AGILE virtual candidate SMILES column."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

CONFIG_SCHEMA_VERSION = "m0_09_agile_virtual_smiles_config.v1"
MANIFEST_SCHEMA_VERSION = "m0_09_agile_virtual_smiles_manifest.v1"
OUTPUT_FIELDS = (
    "source_row_index",
    "source_smiles",
    "canonical_isomeric_smiles",
)


class AgileVirtualExtractionError(ValueError):
    """Raised when the AGILE virtual source violates the frozen contract."""


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    try:
        handle = path.open("rb")
    except FileNotFoundError as exc:
        raise AgileVirtualExtractionError(
            f"AGILE virtual source not found: {path}"
        ) from exc
    with handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AgileVirtualExtractionError(f"config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AgileVirtualExtractionError(f"config is not valid JSON: {exc}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise AgileVirtualExtractionError(
            f"config schema must be {CONFIG_SCHEMA_VERSION!r}"
        )
    if not isinstance(config.get("source"), dict) or not isinstance(
        config.get("expected_counts"), dict
    ):
        raise AgileVirtualExtractionError(
            "config must define source and expected_counts objects"
        )
    return config


def _expect_integer(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgileVirtualExtractionError(f"{label} must be an integer")
    if positive and value <= 0:
        raise AgileVirtualExtractionError(f"{label} must be positive")
    return value


def _canonicalize(smiles: str, *, row_index: int) -> str:
    if not smiles:
        raise AgileVirtualExtractionError(
            f"source row {row_index} has an empty SMILES value"
        )
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise AgileVirtualExtractionError(
            f"source row {row_index} has invalid SMILES"
        )
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )


def _gzip_csv_bytes(rows: list[dict[str, str | int]]) -> bytes:
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=OUTPUT_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(csv_buffer.getvalue().encode())
    return output.getvalue()


def extract_agile_virtual_smiles(
    config_path: Path,
    source_path: Path,
) -> tuple[dict[str, Any], bytes]:
    """Return the manifest and deterministic compressed SMILES-only table."""

    config = _load_config(config_path)
    source = config["source"]
    expected = config["expected_counts"]
    expected_sha256 = source.get("expected_sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise AgileVirtualExtractionError(
            "source expected_sha256 must be a 64-character string"
        )
    observed_sha256 = _sha256_file(source_path)
    if observed_sha256 != expected_sha256:
        raise AgileVirtualExtractionError(
            f"source hash mismatch: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    expected_bytes = _expect_integer(
        source.get("expected_bytes"),
        label="source expected_bytes",
        positive=True,
    )
    if source_path.stat().st_size != expected_bytes:
        raise AgileVirtualExtractionError(
            f"source byte count mismatch: expected {expected_bytes}, "
            f"observed {source_path.stat().st_size}"
        )
    expected_columns = _expect_integer(
        source.get("expected_header_columns"),
        label="source expected_header_columns",
        positive=True,
    )
    smiles_column = source.get("smiles_column")
    if not isinstance(smiles_column, str) or not smiles_column:
        raise AgileVirtualExtractionError(
            "source smiles_column must be a nonempty string"
        )

    rows: list[dict[str, str | int]] = []
    source_smiles: set[str] = set()
    canonical_smiles: set[str] = set()
    try:
        handle = source_path.open(newline="")
        with handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if (
                header is None
                or len(header) != expected_columns
                or header[0] != smiles_column
            ):
                raise AgileVirtualExtractionError(
                    "source header does not match the frozen SMILES-first schema"
                )
            for source_row_index, row in enumerate(reader):
                if len(row) != len(header):
                    raise AgileVirtualExtractionError(
                        f"source row {source_row_index} has {len(row)} columns; "
                        f"expected {len(header)}"
                    )
                smiles = row[0]
                canonical = _canonicalize(
                    smiles,
                    row_index=source_row_index,
                )
                rows.append(
                    {
                        "source_row_index": source_row_index,
                        "source_smiles": smiles,
                        "canonical_isomeric_smiles": canonical,
                    }
                )
                source_smiles.add(smiles)
                canonical_smiles.add(canonical)
    except csv.Error as exc:
        raise AgileVirtualExtractionError(f"source CSV parse failed: {exc}") from exc

    counts = {
        "source_rows": len(rows),
        "valid_smiles": len(rows),
        "unique_source_smiles": len(source_smiles),
        "unique_canonical_smiles": len(canonical_smiles),
    }
    for label, observed in counts.items():
        configured = _expect_integer(expected.get(label), label=label)
        if observed != configured:
            raise AgileVirtualExtractionError(
                f"{label} mismatch: expected {configured}, observed {observed}"
            )

    payload = _gzip_csv_bytes(rows)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task": config["task"],
        "generated_utc": config["generated_utc"],
        "randomness": {"seed": 0, "used": False},
        "source": {
            "asset": source["asset"],
            "doi": source["doi"],
            "bytes": source_path.stat().st_size,
            "sha256": observed_sha256,
            "header_columns": expected_columns,
            "smiles_column": smiles_column,
        },
        "output": {
            "asset": "data/derived/agile_virtual12k_smiles.csv.gz",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "fields": list(OUTPUT_FIELDS),
            "compression": "gzip_mtime_0",
        },
        "summary": counts,
        "claims_boundary": config["claims_boundary"],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
    }
    return manifest, payload


def write_agile_virtual_smiles(
    manifest: dict[str, Any],
    payload: bytes,
    output_path: Path,
    manifest_path: Path,
) -> None:
    """Atomically write the derived table and its provenance manifest."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_descriptor, output_temporary = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    manifest_descriptor, manifest_temporary = tempfile.mkstemp(
        dir=manifest_path.parent,
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
    )
    output_temporary_path = Path(output_temporary)
    manifest_temporary_path = Path(manifest_temporary)
    try:
        with os.fdopen(output_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        with os.fdopen(manifest_descriptor, "wb") as handle:
            handle.write(
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(output_temporary_path, output_path)
        os.replace(manifest_temporary_path, manifest_path)
    except Exception:
        output_temporary_path.unlink(missing_ok=True)
        manifest_temporary_path.unlink(missing_ok=True)
        raise
