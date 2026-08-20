"""Build a hash-verified source package index for M0-09 route review."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from rdkit import rdBase

from forge.synthesis.sources.pmc_sources import load_pmc_queue
from forge.synthesis.sources.supervision_inventory import sha256_file

RESULT_SCHEMA_VERSION = "m0_09_source_review_index.v1"


class SourceReviewError(ValueError):
    """Raised when a source package cannot be verified safely."""


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SourceReviewError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SourceReviewError(f"{label} is not valid JSON: {exc}") from exc


def _verify_asset(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_file():
        raise SourceReviewError(f"{label} is missing from the source cache: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise SourceReviewError(
            f"{label} hash mismatch for {path}: {observed} != {expected_sha256}"
        )


def _safe_member_name(name: str) -> str:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or len(member.parts) != 1 or not member.name:
        raise SourceReviewError(f"unsafe supplementary archive member: {name!r}")
    return member.name


def _asset_kind(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    return {
        ".docx": "narrative_or_structure_document",
        ".mp4": "media",
        ".pdf": "narrative_or_structure_document",
        ".xlsx": "tabular",
        ".zip": "nested_archive",
    }.get(extension, "other")


def _component_source_package_coverage(
    component_ledger_path: Path,
    *,
    complete_pmids: set[str],
    incomplete_pmids: set[str],
) -> dict[str, Any]:
    states = (
        "any_complete_source_package",
        "only_incomplete_pmc_source_package",
        "non_pmc_or_unlinked_source_package",
    )
    counts = {state: 0 for state in states}
    by_role: dict[str, dict[str, int]] = {}
    try:
        handle = component_ledger_path.open(newline="")
    except FileNotFoundError as exc:
        raise SourceReviewError(
            f"component source ledger not found: {component_ledger_path}"
        ) from exc
    with handle:
        reader = csv.DictReader(handle)
        required = {"component_id", "role", "source_pmids_json"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SourceReviewError(
                f"component source ledger must contain columns {sorted(required)}"
            )
        total = 0
        for row in reader:
            total += 1
            try:
                pmids = set(json.loads(row["source_pmids_json"]))
            except json.JSONDecodeError as exc:
                raise SourceReviewError(
                    f"invalid source_pmids_json for {row['component_id']}"
                ) from exc
            if pmids & complete_pmids:
                state = "any_complete_source_package"
            elif pmids & incomplete_pmids:
                state = "only_incomplete_pmc_source_package"
            else:
                state = "non_pmc_or_unlinked_source_package"
            counts[state] += 1
            role_counts = by_role.setdefault(
                row["role"],
                {candidate: 0 for candidate in states},
            )
            role_counts[state] += 1
    if total == 0:
        raise SourceReviewError("component source ledger contains no components")
    return {
        "total_components": total,
        **counts,
        "by_role": {role: by_role[role] for role in sorted(by_role)},
    }


def _extract_official_supplements(
    record: dict[str, Any],
    cache_root: Path,
) -> list[dict[str, Any]]:
    endpoint = record["endpoints"]["europe_pmc_supplements"]
    archive_relative = Path("m0_09_pmc") / endpoint["cache_asset"]
    archive_path = cache_root / archive_relative
    _verify_asset(
        archive_path,
        endpoint["sha256"],
        label=f"{record['pmc_id']} official supplement archive",
    )
    expected_names = record["jats_supplement_links"]
    if not expected_names:
        raise SourceReviewError(
            f"{record['pmc_id']} has an official archive but no JATS supplement links"
        )

    extracted = []
    with zipfile.ZipFile(archive_path) as archive:
        member_names = {member.filename for member in archive.infolist()}
        for expected_name in expected_names:
            filename = _safe_member_name(expected_name)
            if expected_name not in member_names:
                raise SourceReviewError(f"{record['pmc_id']} archive is missing {expected_name!r}")
            content = archive.read(expected_name)
            output_relative = Path("m0_09_review_assets") / record["pmc_id"] / filename
            output_path = cache_root / output_relative
            _atomic_write_bytes(output_path, content)
            extracted.append(
                {
                    "filename": filename,
                    "asset_kind": _asset_kind(filename),
                    "cache_asset": str(output_relative),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "source_channel": "europe_pmc_supplement_archive",
                    "review_status": "not_started",
                }
            )
    return sorted(extracted, key=lambda asset: asset["filename"])


def _publisher_assets(
    record: dict[str, Any],
    cache_root: Path,
) -> list[dict[str, Any]]:
    assets = []
    for asset in record["assets"]:
        if not asset["validated"]:
            raise SourceReviewError(
                f"{record['pmc_id']} publisher asset is not validated: " f"{asset['filename']}"
            )
        relative = Path("m0_09_publishers") / asset["cache_asset"]
        _verify_asset(
            cache_root / relative,
            asset["sha256"],
            label=f"{record['pmc_id']} publisher supplement",
        )
        assets.append(
            {
                "filename": asset["filename"],
                "asset_kind": _asset_kind(asset["filename"]),
                "cache_asset": str(relative),
                "bytes": asset["bytes"],
                "sha256": asset["sha256"],
                "source_channel": "publisher_page",
                "source_url": asset["url"],
                "review_status": "not_started",
            }
        )
    return sorted(assets, key=lambda asset: asset["filename"])


def build_source_review_index(
    source_queue_path: Path,
    component_ledger_path: Path,
    pmc_acquisition_path: Path,
    publisher_acquisition_path: Path,
    cache_root: Path,
    *,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Merge verified main texts and supplementary files into one review queue."""

    queue = load_pmc_queue(source_queue_path)
    pmc = _load_json(pmc_acquisition_path, label="PMC acquisition manifest")
    publisher = _load_json(
        publisher_acquisition_path,
        label="publisher acquisition manifest",
    )
    pmc_records = {record["pmc_id"]: record for record in pmc["records"]}
    publisher_records = {record["pmc_id"]: record for record in publisher["records"]}
    expected_pmc_ids = {entry.pmc_id for entry in queue}
    if set(pmc_records) != expected_pmc_ids:
        raise SourceReviewError("PMC acquisition manifest does not match the 26-source queue")
    if not set(publisher_records).issubset(expected_pmc_ids):
        raise SourceReviewError("publisher manifest contains a PMCID outside the queue")

    records = []
    for entry in queue:
        pmc_record = pmc_records[entry.pmc_id]
        main_endpoint = pmc_record["endpoints"]["bioc_full_text"]
        main_relative = Path("m0_09_pmc") / main_endpoint["cache_asset"]
        _verify_asset(
            cache_root / main_relative,
            main_endpoint["sha256"],
            label=f"{entry.pmc_id} machine-readable main text",
        )
        main_text = {
            "cache_asset": str(main_relative),
            "bytes": main_endpoint["bytes"],
            "sha256": main_endpoint["sha256"],
            "source_url": main_endpoint["url"],
            "review_status": "not_started",
        }

        if entry.pmc_id in publisher_records:
            publisher_record = publisher_records[entry.pmc_id]
            if publisher_record["source_package_status"] != "complete":
                raise SourceReviewError(
                    f"publisher source package is incomplete for {entry.pmc_id}"
                )
            supplements = _publisher_assets(publisher_record, cache_root)
            package_channel = "publisher_page"
            package_status = "complete"
            gap_reason = ""
        elif pmc_record["supplement_archive"]["status"] == "validated_archive_acquired":
            supplements = _extract_official_supplements(pmc_record, cache_root)
            package_channel = "europe_pmc_supplement_archive"
            package_status = "complete"
            gap_reason = ""
        else:
            supplements = []
            package_channel = "unresolved"
            package_status = "supplement_unavailable"
            gap_reason = pmc_record["supplement_status"]

        records.append(
            {
                "review_rank": entry.review_rank,
                "source_id": entry.source_id,
                "pmid": entry.pmid,
                "pmc_id": entry.pmc_id,
                "doi": entry.doi,
                "title": entry.title,
                "unique_component_count": entry.unique_component_count,
                "lnpdb_record_count": entry.lnpdb_record_count,
                "source_package_status": package_status,
                "source_package_channel": package_channel,
                "gap_reason": gap_reason,
                "main_text": main_text,
                "supplement_count": len(supplements),
                "supplements": supplements,
                "route_review_status": "not_started",
            }
        )

    if generated_utc is None:
        generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    else:
        try:
            parsed = dt.datetime.fromisoformat(generated_utc)
        except ValueError as exc:
            raise SourceReviewError(
                f"generated_utc is not valid ISO-8601: {generated_utc!r}"
            ) from exc
        if parsed.tzinfo is None:
            raise SourceReviewError("generated_utc must include a timezone")
        generated = parsed.isoformat()

    packages_complete = sum(record["source_package_status"] == "complete" for record in records)
    supplements = sum(record["supplement_count"] for record in records)
    complete_pmids = {
        record["pmid"] for record in records if record["source_package_status"] == "complete"
    }
    incomplete_pmids = {
        record["pmid"] for record in records if record["source_package_status"] != "complete"
    }
    component_coverage = _component_source_package_coverage(
        component_ledger_path,
        complete_pmids=complete_pmids,
        incomplete_pmids=incomplete_pmids,
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-09 hash-verified source review index",
        "generated_utc": generated,
        "randomness": {"seed": 0, "used": False},
        "inputs": [
            {
                "asset": _portable_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                source_queue_path,
                component_ledger_path,
                pmc_acquisition_path,
                publisher_acquisition_path,
            )
        ],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "summary": {
            "pmc_publication_families": len(records),
            "source_packages_complete": packages_complete,
            "source_packages_supplement_unavailable": len(records) - packages_complete,
            "machine_readable_main_texts": len(records),
            "supplementary_review_assets": supplements,
            "route_reviews_completed": 0,
        },
        "component_source_package_coverage": component_coverage,
        "evidence_contract": {
            "source_package_complete": "Main text and all publisher-listed or JATS-listed attachments are local and hash-verified.",
            "supplement_unavailable": "Main text is local, but the supplementary package has not been acquired.",
            "review_not_started": "No chemistry or route claim has yet been extracted from the source package.",
        },
        "records": records,
    }


def write_source_review_index(result: dict[str, Any], output_path: Path) -> None:
    """Atomically write the source review index."""

    _atomic_write_bytes(
        output_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
