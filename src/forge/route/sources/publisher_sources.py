"""Acquire publisher-hosted supplementary assets for the M0-09 source review."""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import os
import platform
import re
import tempfile
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rdkit import rdBase

from forge.route.sources.pmc_sources import FetchResponse, fetch_url, load_pmc_queue
from forge.route.sources.supervision_inventory import sha256_file

RESULT_SCHEMA_VERSION = "m0_09_publisher_source_acquisition.v1"
CONFIG_SCHEMA_VERSION = "m0_09_publisher_sources.v1"
ALLOWED_PAGE_HOSTS = {
    "www.nature.com",
    "jnanobiotechnology.biomedcentral.com",
}
ASSET_HOST = "media.springernature.com"
ASSET_PATH_FRAGMENT = "/springer-static/esm/"
ASSET_EXTENSIONS = {".docx", ".mp4", ".pdf", ".xlsx", ".zip"}
ASSET_URL_PATTERN = re.compile(rb"https://media\.springernature\.com/original/[^\"'<>\s]+")


class PublisherSourceError(ValueError):
    """Raised when publisher-source configuration violates its evidence contract."""


@dataclass(frozen=True)
class PublisherSource:
    """One publisher page with a frozen supplementary-asset count."""

    pmc_id: str
    doi: str
    publisher: str
    publisher_page_url: str
    expected_asset_count: int


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


def load_publisher_sources(
    config_path: Path,
    source_queue_path: Path,
) -> list[PublisherSource]:
    """Load and cross-check the publisher-source configuration."""

    try:
        raw = json.loads(config_path.read_text())
    except FileNotFoundError as exc:
        raise PublisherSourceError(f"publisher-source config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise PublisherSourceError(f"publisher-source config is not valid JSON: {exc}") from exc

    if raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise PublisherSourceError(
            f"expected config schema {CONFIG_SCHEMA_VERSION!r}, "
            f"found {raw.get('schema_version')!r}"
        )
    records = raw.get("records")
    if not isinstance(records, list) or not records:
        raise PublisherSourceError("publisher-source config must contain records")

    queue = {entry.pmc_id: entry for entry in load_pmc_queue(source_queue_path)}
    sources = []
    for record in records:
        try:
            source = PublisherSource(
                pmc_id=str(record["pmc_id"]),
                doi=str(record["doi"]),
                publisher=str(record["publisher"]),
                publisher_page_url=str(record["publisher_page_url"]),
                expected_asset_count=int(record["expected_asset_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PublisherSourceError(f"invalid publisher-source record: {record!r}") from exc
        queue_entry = queue.get(source.pmc_id)
        if queue_entry is None:
            raise PublisherSourceError(
                f"publisher-source PMCID is not in the M0-09 queue: {source.pmc_id}"
            )
        if queue_entry.doi.lower() != source.doi.lower():
            raise PublisherSourceError(
                f"DOI mismatch for {source.pmc_id}: " f"{source.doi!r} != {queue_entry.doi!r}"
            )
        page = urlparse(source.publisher_page_url)
        if page.scheme != "https" or page.hostname not in ALLOWED_PAGE_HOSTS:
            raise PublisherSourceError(
                f"publisher page is outside the allowlist: {source.publisher_page_url}"
            )
        if source.expected_asset_count < 1:
            raise PublisherSourceError(f"expected_asset_count must be positive for {source.pmc_id}")
        sources.append(source)

    if len({source.pmc_id for source in sources}) != len(sources):
        raise PublisherSourceError("publisher-source config contains duplicate PMCIDs")
    return sorted(sources, key=lambda source: source.pmc_id)


def discover_asset_urls(page_content: bytes) -> list[str]:
    """Extract allowlisted supplementary-asset URLs from publisher HTML."""

    discovered = set()
    for raw_url in ASSET_URL_PATTERN.findall(page_content):
        url = html.unescape(raw_url.decode())
        parsed = urlparse(url)
        extension = Path(parsed.path).suffix.lower()
        if (
            parsed.scheme == "https"
            and parsed.hostname == ASSET_HOST
            and ASSET_PATH_FRAGMENT in parsed.path
            and extension in ASSET_EXTENSIONS
        ):
            discovered.add(url)
    return sorted(discovered)


def _validate_asset(url: str, content: bytes) -> str:
    extension = Path(urlparse(url).path).suffix.lower()
    if extension == ".pdf":
        return "" if content.startswith(b"%PDF-") else "missing PDF signature"
    if extension in {".docx", ".xlsx", ".zip"}:
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                bad_member = archive.testzip()
        except zipfile.BadZipFile:
            return "invalid ZIP container"
        return f"CRC check failed for {bad_member}" if bad_member else ""
    if extension == ".mp4":
        return "" if len(content) >= 12 and content[4:8] == b"ftyp" else "invalid MP4 signature"
    return f"unsupported asset extension: {extension}"


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


def acquire_publisher_source(
    source: PublisherSource,
    cache_dir: Path,
    *,
    fetch: FetchFunction = fetch_url,
) -> dict[str, Any]:
    """Acquire one publisher page and every discovered supplementary asset."""

    source_dir = cache_dir / source.pmc_id
    page_response = fetch(source.publisher_page_url)
    page_record = _response_record(
        page_response,
        cache_path=source_dir / "publisher_page.html",
        cache_asset=f"{source.pmc_id}/publisher_page.html",
    )
    asset_urls = (
        discover_asset_urls(page_response.content)
        if page_response.status == 200 and page_response.content
        else []
    )
    discovery_matches_expected = len(asset_urls) == source.expected_asset_count
    assets = []
    for url in asset_urls:
        response = fetch(url)
        filename = Path(urlparse(url).path).name
        record = _response_record(
            response,
            cache_path=source_dir / "assets" / filename,
            cache_asset=f"{source.pmc_id}/assets/{filename}",
        )
        record["filename"] = filename
        record["validation_error"] = (
            _validate_asset(url, response.content)
            if response.status == 200 and response.content
            else ""
        )
        record["validated"] = bool(record["cache_asset"]) and not record["validation_error"]
        assets.append(record)

    validated_assets = sum(asset["validated"] for asset in assets)
    complete = (
        bool(page_record["cache_asset"])
        and discovery_matches_expected
        and validated_assets == source.expected_asset_count
    )
    return {
        "pmc_id": source.pmc_id,
        "doi": source.doi,
        "publisher": source.publisher,
        "publisher_page": page_record,
        "expected_asset_count": source.expected_asset_count,
        "discovered_asset_count": len(asset_urls),
        "discovery_matches_expected": discovery_matches_expected,
        "validated_asset_count": validated_assets,
        "assets": assets,
        "source_package_status": "complete" if complete else "incomplete",
        "route_review_status": "not_started",
    }


def acquire_publisher_sources(
    config_path: Path,
    source_queue_path: Path,
    cache_dir: Path,
    *,
    generated_utc: str | None = None,
    workers: int = 4,
    fetch: FetchFunction = fetch_url,
) -> dict[str, Any]:
    """Acquire the configured publisher source packages with bounded concurrency."""

    if workers < 1:
        raise PublisherSourceError("workers must be at least 1")
    sources = load_publisher_sources(config_path, source_queue_path)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                acquire_publisher_source,
                source,
                cache_dir,
                fetch=fetch,
            ): source
            for source in sources
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda record: record["pmc_id"])

    if generated_utc is None:
        generated = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    else:
        try:
            parsed = dt.datetime.fromisoformat(generated_utc)
        except ValueError as exc:
            raise PublisherSourceError(
                f"generated_utc is not valid ISO-8601: {generated_utc!r}"
            ) from exc
        if parsed.tzinfo is None:
            raise PublisherSourceError("generated_utc must include a timezone")
        generated = parsed.isoformat()

    packages_complete = sum(record["source_package_status"] == "complete" for record in records)
    expected_assets = sum(record["expected_asset_count"] for record in records)
    validated_assets = sum(record["validated_asset_count"] for record in records)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-09 publisher supplementary-source acquisition",
        "generated_utc": generated,
        "randomness": {"seed": 0, "used": False},
        "inputs": [
            {
                "asset": _portable_path(config_path),
                "bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            },
            {
                "asset": _portable_path(source_queue_path),
                "bytes": source_queue_path.stat().st_size,
                "sha256": sha256_file(source_queue_path),
            },
        ],
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "platform": platform.platform(),
        },
        "parameters": {"workers": workers},
        "summary": {
            "configured_publication_families": len(records),
            "source_packages_complete": packages_complete,
            "source_packages_incomplete": len(records) - packages_complete,
            "expected_assets": expected_assets,
            "validated_assets": validated_assets,
            "complete": packages_complete == len(records) and validated_assets == expected_assets,
            "route_reviews_completed": 0,
        },
        "evidence_contract": {
            "publisher_page_acquired": "The exact configured publisher page was downloaded and hashed.",
            "asset_discovered": "The attachment URL appeared on that downloaded publisher page.",
            "asset_validated": "The attachment was downloaded, hashed, and passed a format check.",
            "route_reviewed": "Every chemistry-relevant attachment was inspected.",
        },
        "records": records,
    }


def write_publisher_acquisition_result(
    result: dict[str, Any],
    output_path: Path,
) -> None:
    """Atomically write the publisher acquisition manifest."""

    _atomic_write_bytes(
        output_path,
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
