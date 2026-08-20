"""Acquire and hash primary-source text for the M0-09 PMC review queue.

PMC indexing, machine-readable full text, open-access archives, and supplementary
attachments are separate facts. This module records each fact independently so an
article cannot be marked route-reviewed merely because a PMCID exists.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import platform
import tarfile
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rdkit import rdBase

from forge.synthesis.sources.supervision_inventory import sha256_file

RESULT_SCHEMA_VERSION = "m0_09_pmc_source_acquisition.v1"
USER_AGENT = "FORGE-M0-09-source-inventory/1.0 (academic route evidence audit)"
XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
MAX_RESPONSE_BYTES = 512 * 1024 * 1024


class PmcSourceError(ValueError):
    """Raised when PMC acquisition violates the source-inventory contract."""


@dataclass(frozen=True)
class FetchResponse:
    """One HTTP response retained for deterministic source accounting."""

    url: str
    status: int | None
    content: bytes
    content_type: str
    transport_error: str


@dataclass(frozen=True)
class PmcQueueEntry:
    """One publication family selected from the LNPDB source queue."""

    review_rank: int
    source_id: str
    pmid: str
    pmc_id: str
    doi: str
    title: str
    unique_component_count: int
    lnpdb_record_count: int


FetchFunction = Callable[[str], FetchResponse]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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


def fetch_url(url: str, *, timeout_seconds: float = 60.0) -> FetchResponse:
    """Fetch one primary-source URL and retain HTTP errors as evidence states."""

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/xml,text/xml,application/json,*/*;q=0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content = response.read(MAX_RESPONSE_BYTES + 1)
            if len(content) > MAX_RESPONSE_BYTES:
                return FetchResponse(
                    url=url,
                    status=response.status,
                    content=b"",
                    content_type=response.headers.get_content_type(),
                    transport_error=(f"response exceeded {MAX_RESPONSE_BYTES}-byte safety limit"),
                )
            return FetchResponse(
                url=url,
                status=response.status,
                content=content,
                content_type=response.headers.get_content_type(),
                transport_error="",
            )
    except urllib.error.HTTPError as exc:
        return FetchResponse(
            url=url,
            status=exc.code,
            content=exc.read(),
            content_type=exc.headers.get_content_type() if exc.headers else "",
            transport_error="",
        )
    except urllib.error.URLError as exc:
        return FetchResponse(
            url=url,
            status=None,
            content=b"",
            content_type="",
            transport_error=str(exc.reason),
        )


def load_pmc_queue(path: Path) -> list[PmcQueueEntry]:
    """Load every PMCID-bearing source in review-rank order."""

    required = {
        "review_rank",
        "source_id",
        "pmid",
        "pmc_id",
        "doi",
        "title",
        "unique_component_count",
        "lnpdb_record_count",
    }
    try:
        handle = path.open(newline="")
    except FileNotFoundError as exc:
        raise PmcSourceError(f"LNPDB source queue not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise PmcSourceError(f"LNPDB source queue must contain columns {sorted(required)}")
        entries = [
            PmcQueueEntry(
                review_rank=int(row["review_rank"]),
                source_id=row["source_id"],
                pmid=row["pmid"],
                pmc_id=row["pmc_id"],
                doi=row["doi"],
                title=row["title"],
                unique_component_count=int(row["unique_component_count"]),
                lnpdb_record_count=int(row["lnpdb_record_count"]),
            )
            for row in reader
            if row["pmc_id"].strip()
        ]

    entries.sort(key=lambda entry: (entry.review_rank, entry.pmc_id))
    if len(entries) != 26:
        raise PmcSourceError(f"expected 26 PMC-linked publication families, found {len(entries)}")
    if len({entry.pmc_id for entry in entries}) != len(entries):
        raise PmcSourceError("PMC source queue contains duplicate PMC identifiers")
    if any(not entry.pmc_id.startswith("PMC") for entry in entries):
        raise PmcSourceError("every PMC identifier must begin with 'PMC'")
    return entries


def _endpoint_urls(pmc_id: str) -> dict[str, str]:
    return {
        "bioc_full_text": (
            "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/"
            f"pmcoa.cgi/BioC_xml/{pmc_id}/unicode"
        ),
        "europe_pmc_jats": (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc_id}/fullTextXML"
        ),
        "pmc_oa_record": (f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}"),
        "europe_pmc_supplements": (
            f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc_id}/supplementaryFiles"
        ),
    }


def _xml_root(response: FetchResponse, *, endpoint: str) -> ET.Element | None:
    if response.status != 200 or not response.content:
        return None
    try:
        return ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise PmcSourceError(
            f"{endpoint} returned malformed XML for {response.url}: {exc}"
        ) from exc


def _jats_supplement_links(root: ET.Element | None) -> list[str]:
    if root is None:
        return []
    links: set[str] = set()
    for element in root.iter():
        local_name = element.tag.rsplit("}", 1)[-1]
        if local_name not in {"supplementary-material", "media"}:
            continue
        href = element.attrib.get(XLINK_HREF) or element.attrib.get("href")
        if href:
            links.add(href.strip())
    return sorted(link for link in links if link)


def _oa_record(root: ET.Element | None) -> dict[str, Any]:
    if root is None:
        return {
            "availability": "http_unavailable",
            "error_code": "",
            "package_links": [],
        }
    error = root.find(".//error")
    if error is not None:
        return {
            "availability": "api_reported_unavailable",
            "error_code": error.attrib.get("code", ""),
            "package_links": [],
        }
    record = root.find(".//record")
    links = []
    if record is not None:
        for link in record.findall(".//link"):
            href = link.attrib.get("href", "").strip()
            if href:
                links.append(
                    {
                        "format": link.attrib.get("format", ""),
                        "href": href,
                        "updated": link.attrib.get("updated", ""),
                    }
                )
    return {
        "availability": "package_link_reported" if links else "no_package_link",
        "error_code": "",
        "license": record.attrib.get("license", "") if record is not None else "",
        "package_links": sorted(links, key=lambda value: (value["format"], value["href"])),
    }


def _oa_package_https_url(oa: dict[str, Any]) -> str:
    for link in oa["package_links"]:
        if link["format"] != "tgz":
            continue
        parsed = urlparse(link["href"])
        if parsed.hostname != "ftp.ncbi.nlm.nih.gov":
            continue
        return parsed._replace(scheme="https").geturl()
    return ""


def _response_record(
    response: FetchResponse,
    *,
    cache_path: Path,
    cache_asset: str,
) -> dict[str, Any]:
    if response.status == 200 and response.content:
        _atomic_write_bytes(cache_path, response.content)
        return {
            "url": response.url,
            "http_status": response.status,
            "content_type": response.content_type,
            "bytes": len(response.content),
            "sha256": _sha256_bytes(response.content),
            "cache_asset": cache_asset,
            "transport_error": "",
        }
    return {
        "url": response.url,
        "http_status": response.status,
        "content_type": response.content_type,
        "bytes": len(response.content),
        "sha256": _sha256_bytes(response.content) if response.content else "",
        "cache_asset": "",
        "transport_error": response.transport_error,
    }


def _tar_archive_record(response: FetchResponse | None) -> dict[str, Any]:
    if response is None:
        return {
            "status": "not_attempted",
            "archive_entries": [],
            "archive_entry_count": 0,
            "uncompressed_bytes": 0,
            "validation_error": "",
        }
    if response.status != 200 or not response.content:
        return {
            "status": "fetch_failed",
            "archive_entries": [],
            "archive_entry_count": 0,
            "uncompressed_bytes": 0,
            "validation_error": response.transport_error,
        }
    try:
        with tarfile.open(fileobj=BytesIO(response.content), mode="r:gz") as archive:
            entries = [
                {
                    "name": member.name,
                    "bytes": member.size,
                    "is_file": member.isfile(),
                }
                for member in archive.getmembers()
            ]
    except (tarfile.TarError, EOFError) as exc:
        return {
            "status": "invalid_archive",
            "archive_entries": [],
            "archive_entry_count": 0,
            "uncompressed_bytes": 0,
            "validation_error": str(exc),
        }
    return {
        "status": "validated_archive_acquired",
        "archive_entries": entries,
        "archive_entry_count": len(entries),
        "uncompressed_bytes": sum(entry["bytes"] for entry in entries),
        "validation_error": "",
    }


def _supplement_archive_record(response: FetchResponse) -> dict[str, Any]:
    if response.status != 200 or not response.content:
        return {
            "status": "not_acquired",
            "archive_entries": [],
            "archive_entry_count": 0,
            "uncompressed_bytes": 0,
            "api_error": "",
            "validation_error": "",
        }
    try:
        with zipfile.ZipFile(BytesIO(response.content)) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise zipfile.BadZipFile(f"CRC check failed for {bad_member}")
            entries = [
                {
                    "name": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "is_directory": info.is_dir(),
                }
                for info in archive.infolist()
            ]
    except zipfile.BadZipFile as exc:
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError:
            root = None
        error_message = root.findtext(".//errMsg", default="") if root is not None else ""
        if error_message:
            return {
                "status": "api_reported_unavailable",
                "archive_entries": [],
                "archive_entry_count": 0,
                "uncompressed_bytes": 0,
                "api_error": error_message,
                "validation_error": "",
            }
        return {
            "status": "invalid_archive",
            "archive_entries": [],
            "archive_entry_count": 0,
            "uncompressed_bytes": 0,
            "api_error": "",
            "validation_error": str(exc),
        }
    return {
        "status": "validated_archive_acquired",
        "archive_entries": entries,
        "archive_entry_count": len(entries),
        "uncompressed_bytes": sum(entry["bytes"] for entry in entries),
        "api_error": "",
        "validation_error": "",
    }


def acquire_pmc_source(
    entry: PmcQueueEntry,
    cache_dir: Path,
    *,
    fetch: FetchFunction = fetch_url,
) -> dict[str, Any]:
    """Acquire official machine-readable records for one PMCID."""

    urls = _endpoint_urls(entry.pmc_id)
    responses = {name: fetch(url) for name, url in urls.items()}
    source_dir = cache_dir / entry.pmc_id
    jats_root = _xml_root(responses["europe_pmc_jats"], endpoint="Europe PMC JATS")
    oa_root = _xml_root(responses["pmc_oa_record"], endpoint="PMC OA API")
    supplement_links = _jats_supplement_links(jats_root)
    oa = _oa_record(oa_root)
    supplement_archive = _supplement_archive_record(responses["europe_pmc_supplements"])
    oa_package_url = _oa_package_https_url(oa)
    oa_package_response = None
    if supplement_archive["status"] != "validated_archive_acquired" and oa_package_url:
        oa_package_response = fetch(oa_package_url)
    oa_package_archive = _tar_archive_record(oa_package_response)

    if supplement_archive["status"] == "validated_archive_acquired":
        supplement_cache_path = source_dir / "supplementary_files.zip"
        supplement_cache_asset = f"{entry.pmc_id}/supplementary_files.zip"
    else:
        supplement_cache_path = source_dir / "supplementary_response.xml"
        supplement_cache_asset = f"{entry.pmc_id}/supplementary_response.xml"
    if oa_package_response is not None:
        if oa_package_archive["status"] == "validated_archive_acquired":
            oa_package_cache_path = source_dir / "oa_package.tar.gz"
            oa_package_cache_asset = f"{entry.pmc_id}/oa_package.tar.gz"
        else:
            oa_package_cache_path = source_dir / "oa_package_response.bin"
            oa_package_cache_asset = f"{entry.pmc_id}/oa_package_response.bin"
        oa_package_endpoint = _response_record(
            oa_package_response,
            cache_path=oa_package_cache_path,
            cache_asset=oa_package_cache_asset,
        )
        oa_package_endpoint["attempted"] = True
    else:
        oa_package_endpoint = {
            "url": oa_package_url,
            "http_status": None,
            "content_type": "",
            "bytes": 0,
            "sha256": "",
            "cache_asset": "",
            "transport_error": "",
            "attempted": False,
        }
    endpoint_records = {
        "bioc_full_text": _response_record(
            responses["bioc_full_text"],
            cache_path=source_dir / "bioc_full_text.xml",
            cache_asset=f"{entry.pmc_id}/bioc_full_text.xml",
        ),
        "europe_pmc_jats": _response_record(
            responses["europe_pmc_jats"],
            cache_path=source_dir / "europe_pmc_jats.xml",
            cache_asset=f"{entry.pmc_id}/europe_pmc_jats.xml",
        ),
        "pmc_oa_record": _response_record(
            responses["pmc_oa_record"],
            cache_path=source_dir / "pmc_oa_record.xml",
            cache_asset=f"{entry.pmc_id}/pmc_oa_record.xml",
        ),
        "europe_pmc_supplements": _response_record(
            responses["europe_pmc_supplements"],
            cache_path=supplement_cache_path,
            cache_asset=supplement_cache_asset,
        ),
        "pmc_oa_package": oa_package_endpoint,
    }

    if endpoint_records["bioc_full_text"]["cache_asset"]:
        main_text_status = "machine_readable_full_text_acquired"
    elif endpoint_records["europe_pmc_jats"]["cache_asset"]:
        main_text_status = "machine_readable_full_text_acquired"
    else:
        main_text_status = "machine_readable_full_text_unavailable"

    if supplement_archive["status"] == "validated_archive_acquired":
        supplement_status = "official_archive_acquired"
    elif oa_package_archive["status"] == "validated_archive_acquired":
        supplement_status = "oa_package_acquired"
    elif oa_package_archive["status"] == "invalid_archive":
        supplement_status = "oa_package_invalid"
    elif oa_package_archive["status"] == "fetch_failed":
        supplement_status = "oa_package_fetch_failed"
    elif supplement_archive["status"] == "api_reported_unavailable":
        supplement_status = "official_api_reported_unavailable"
    elif supplement_archive["status"] == "invalid_archive":
        supplement_status = "official_archive_invalid"
    elif supplement_links:
        supplement_status = "jats_links_discovered_not_downloaded"
    elif oa["package_links"]:
        supplement_status = "oa_package_reported_not_downloaded"
    else:
        supplement_status = "publisher_discovery_required"

    return {
        "review_rank": entry.review_rank,
        "source_id": entry.source_id,
        "pmid": entry.pmid,
        "pmc_id": entry.pmc_id,
        "doi": entry.doi,
        "title": entry.title,
        "unique_component_count": entry.unique_component_count,
        "lnpdb_record_count": entry.lnpdb_record_count,
        "main_text_status": main_text_status,
        "supplement_status": supplement_status,
        "route_review_status": "not_started",
        "endpoints": endpoint_records,
        "jats_supplement_links": supplement_links,
        "oa_record": oa,
        "supplement_archive": supplement_archive,
        "oa_package_archive": oa_package_archive,
    }


def acquire_pmc_queue(
    source_queue_path: Path,
    cache_dir: Path,
    *,
    generated_utc: str | None = None,
    workers: int = 6,
    fetch: FetchFunction = fetch_url,
) -> dict[str, Any]:
    """Acquire every PMC-linked source with bounded parallel network use."""

    if workers < 1:
        raise PmcSourceError("workers must be at least 1")
    entries = load_pmc_queue(source_queue_path)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(acquire_pmc_source, entry, cache_dir, fetch=fetch): entry
            for entry in entries
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: (record["review_rank"], record["pmc_id"]))

    acquired = sum(
        record["main_text_status"] == "machine_readable_full_text_acquired" for record in records
    )
    if acquired == 0:
        raise PmcSourceError(
            "no machine-readable PMC full text was acquired; check network and endpoint access"
        )

    if generated_utc is None:
        generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    else:
        try:
            parsed = dt.datetime.fromisoformat(generated_utc)
        except ValueError as exc:
            raise PmcSourceError(f"generated_utc is not valid ISO-8601: {generated_utc!r}") from exc
        if parsed.tzinfo is None:
            raise PmcSourceError("generated_utc must include a timezone")
        generated = parsed.isoformat()

    supplement_counts: dict[str, int] = {}
    for record in records:
        status = record["supplement_status"]
        supplement_counts[status] = supplement_counts.get(status, 0) + 1

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-09 PMC primary-source acquisition",
        "generated_utc": generated,
        "randomness": {"seed": 0, "used": False},
        "inputs": [
            {
                "asset": _portable_path(source_queue_path),
                "bytes": source_queue_path.stat().st_size,
                "sha256": sha256_file(source_queue_path),
            }
        ],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "parameters": {
            "workers": workers,
            "user_agent": USER_AGENT,
            "max_response_bytes": MAX_RESPONSE_BYTES,
        },
        "summary": {
            "pmc_publication_families": len(records),
            "machine_readable_full_text_acquired": acquired,
            "machine_readable_full_text_unavailable": len(records) - acquired,
            "jats_full_text_acquired": sum(
                record["endpoints"]["europe_pmc_jats"]["http_status"] == 200 for record in records
            ),
            "oa_package_links_reported": sum(
                bool(record["oa_record"]["package_links"]) for record in records
            ),
            "supplement_archives_acquired": sum(
                record["supplement_archive"]["status"] == "validated_archive_acquired"
                for record in records
            ),
            "supplement_archive_entries": sum(
                record["supplement_archive"]["archive_entry_count"] for record in records
            ),
            "oa_packages_acquired": sum(
                record["oa_package_archive"]["status"] == "validated_archive_acquired"
                for record in records
            ),
            "oa_package_entries": sum(
                record["oa_package_archive"]["archive_entry_count"] for record in records
            ),
            "supplement_status_counts": {
                key: supplement_counts[key] for key in sorted(supplement_counts)
            },
            "route_reviews_completed": 0,
        },
        "evidence_contract": {
            "pmc_indexed": "A PMCID exists.",
            "main_text_acquired": "Exact machine-readable article text was downloaded and hashed.",
            "supplement_discovered": "A JATS or OA-package link was found but not necessarily downloaded.",
            "route_reviewed": "Main text and every relevant supplementary attachment were inspected.",
            "route_extracted": "An exact source-located reaction record was curated.",
        },
        "records": records,
    }


def write_acquisition_result(result: dict[str, Any], output_path: Path) -> None:
    """Atomically write the deterministic acquisition manifest."""

    _atomic_write_bytes(
        output_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
