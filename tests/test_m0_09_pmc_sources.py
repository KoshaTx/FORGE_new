from __future__ import annotations

import csv
import json
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from forge.synthesis.sources.pmc_sources import (
    FetchResponse,
    PmcSourceError,
    acquire_pmc_queue,
    load_pmc_queue,
    write_acquisition_result,
)


def _write_source_queue(path: Path, count: int = 26) -> None:
    fields = [
        "review_rank",
        "source_id",
        "pmid",
        "pmc_id",
        "doi",
        "title",
        "unique_component_count",
        "lnpdb_record_count",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(1, count + 1):
            writer.writerow(
                {
                    "review_rank": index,
                    "source_id": str(1000 + index),
                    "pmid": str(1000 + index),
                    "pmc_id": f"PMC{index}",
                    "doi": f"10.test/{index}",
                    "title": f"Paper {index}",
                    "unique_component_count": count - index,
                    "lnpdb_record_count": index,
                }
            )


def _fake_fetch(url: str) -> FetchResponse:
    if "fullTextXML" in url:
        content = (
            b'<article xmlns:xlink="http://www.w3.org/1999/xlink">'
            b'<supplementary-material xlink:href="supp/file.pdf"/>'
            b"</article>"
        )
        content_type = "application/xml"
    elif "supplementaryFiles" in url:
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("supp/file.pdf", b"synthetic PDF bytes")
        content = buffer.getvalue()
        content_type = "application/zip"
    elif "oa.fcgi" in url:
        content = (
            b'<OA><records><record license="CC BY">'
            b'<link format="tgz" href="ftp://example/package.tgz" updated="2026"/>'
            b"</record></records></OA>"
        )
        content_type = "application/xml"
    else:
        content = b"<collection><document><id>test</id></document></collection>"
        content_type = "application/xml"
    return FetchResponse(
        url=url,
        status=200,
        content=content,
        content_type=content_type,
        transport_error="",
    )


def test_load_queue_requires_all_26_unique_pmc_sources(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    _write_source_queue(queue)
    entries = load_pmc_queue(queue)

    assert len(entries) == 26
    assert entries[0].pmc_id == "PMC1"
    assert entries[-1].pmc_id == "PMC26"

    _write_source_queue(queue, count=25)
    with pytest.raises(PmcSourceError, match="expected 26"):
        load_pmc_queue(queue)


def test_acquisition_separates_main_text_supplement_and_route_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.csv"
    _write_source_queue(queue)
    monkeypatch.chdir(tmp_path)

    result = acquire_pmc_queue(
        queue,
        tmp_path / "cache",
        generated_utc="2026-07-29T00:00:00+00:00",
        workers=2,
        fetch=_fake_fetch,
    )

    assert result["summary"]["machine_readable_full_text_acquired"] == 26
    assert result["summary"]["jats_full_text_acquired"] == 26
    assert result["summary"]["oa_package_links_reported"] == 26
    assert result["summary"]["supplement_archives_acquired"] == 26
    assert result["summary"]["supplement_archive_entries"] == 26
    assert result["summary"]["route_reviews_completed"] == 0
    assert result["inputs"][0]["asset"] == "queue.csv"
    assert all(record["route_review_status"] == "not_started" for record in result["records"])
    assert all(record["jats_supplement_links"] == ["supp/file.pdf"] for record in result["records"])
    assert all(
        record["supplement_status"] == "official_archive_acquired" for record in result["records"]
    )
    assert result["records"][0]["endpoints"]["bioc_full_text"]["cache_asset"] == (
        "PMC1/bioc_full_text.xml"
    )


def test_transport_failure_does_not_masquerade_as_source_absence(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.csv"
    _write_source_queue(queue)

    def partly_failed_fetch(url: str) -> FetchResponse:
        if "BioC_xml" in url:
            return _fake_fetch(url)
        return FetchResponse(
            url=url,
            status=None,
            content=b"",
            content_type="",
            transport_error="temporary DNS failure",
        )

    result = acquire_pmc_queue(
        queue,
        tmp_path / "cache",
        generated_utc="2026-07-29T00:00:00+00:00",
        workers=1,
        fetch=partly_failed_fetch,
    )

    record = result["records"][0]
    assert record["main_text_status"] == "machine_readable_full_text_acquired"
    assert record["supplement_status"] == "publisher_discovery_required"
    assert record["endpoints"]["europe_pmc_jats"]["transport_error"] == "temporary DNS failure"


def test_supplement_api_error_is_not_recorded_as_an_archive(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    _write_source_queue(queue)

    def unavailable_supplement_fetch(url: str) -> FetchResponse:
        if "supplementaryFiles" not in url:
            return _fake_fetch(url)
        return FetchResponse(
            url=url,
            status=200,
            content=(
                b"<?xml version='1.0'?><errorBean><errCode>0</errCode>"
                b"<errMsg>Article is not open access</errMsg></errorBean>"
            ),
            content_type="application/xml",
            transport_error="",
        )

    result = acquire_pmc_queue(
        queue,
        tmp_path / "cache",
        generated_utc="2026-07-29T00:00:00+00:00",
        workers=1,
        fetch=unavailable_supplement_fetch,
    )

    record = result["records"][0]
    assert record["supplement_status"] == "official_api_reported_unavailable"
    assert record["supplement_archive"]["api_error"] == "Article is not open access"
    assert record["endpoints"]["europe_pmc_supplements"]["cache_asset"].endswith(
        "supplementary_response.xml"
    )
    assert result["summary"]["supplement_archives_acquired"] == 0


def test_ncbi_oa_package_is_used_as_an_official_fallback(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    _write_source_queue(queue)
    tar_buffer = BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as archive:
        payload = b"supplement contents"
        info = tarfile.TarInfo(name="paper/supplement.pdf")
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))
    package_bytes = tar_buffer.getvalue()

    def oa_fallback_fetch(url: str) -> FetchResponse:
        if "supplementaryFiles" in url:
            return FetchResponse(
                url=url,
                status=404,
                content=b"",
                content_type="text/plain",
                transport_error="",
            )
        if url == "https://ftp.ncbi.nlm.nih.gov/pub/pmc/package.tar.gz":
            return FetchResponse(
                url=url,
                status=200,
                content=package_bytes,
                content_type="application/gzip",
                transport_error="",
            )
        if "oa.fcgi" in url:
            return FetchResponse(
                url=url,
                status=200,
                content=(
                    b"<OA><records><record license='CC BY'>"
                    b"<link format='tgz' "
                    b"href='ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/package.tar.gz'/>"
                    b"</record></records></OA>"
                ),
                content_type="application/xml",
                transport_error="",
            )
        return _fake_fetch(url)

    result = acquire_pmc_queue(
        queue,
        tmp_path / "cache",
        generated_utc="2026-07-29T00:00:00+00:00",
        workers=1,
        fetch=oa_fallback_fetch,
    )

    record = result["records"][0]
    assert record["supplement_status"] == "oa_package_acquired"
    assert record["oa_package_archive"]["archive_entry_count"] == 1
    assert record["endpoints"]["pmc_oa_package"]["attempted"] is True
    assert record["endpoints"]["pmc_oa_package"]["cache_asset"] == ("PMC1/oa_package.tar.gz")
    assert result["summary"]["oa_packages_acquired"] == 26


def test_acquisition_manifest_write_is_byte_stable(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    result = {
        "schema_version": "test",
        "generated_utc": "2026-07-29T00:00:00+00:00",
        "summary": {"records": 26},
    }

    write_acquisition_result(result, output)
    first = output.read_bytes()
    write_acquisition_result(result, output)

    assert output.read_bytes() == first
    assert json.loads(first) == result
