from __future__ import annotations

import csv
import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from forge.route.pmc_sources import FetchResponse
from forge.route.publisher_sources import (
    PublisherSourceError,
    acquire_publisher_sources,
    load_publisher_sources,
    write_publisher_acquisition_result,
)


def _write_source_queue(path: Path) -> None:
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
        for index in range(1, 27):
            writer.writerow(
                {
                    "review_rank": index,
                    "source_id": str(1000 + index),
                    "pmid": str(1000 + index),
                    "pmc_id": f"PMC{index}",
                    "doi": f"10.test/{index}",
                    "title": f"Paper {index}",
                    "unique_component_count": 26 - index,
                    "lnpdb_record_count": index,
                }
            )


def _write_config(
    path: Path,
    *,
    page_url: str = "https://www.nature.com/articles/test-1",
    expected_assets: int = 2,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "m0_09_publisher_sources.v1",
                "records": [
                    {
                        "pmc_id": "PMC1",
                        "doi": "10.test/1",
                        "publisher": "Test Publisher",
                        "publisher_page_url": page_url,
                        "expected_asset_count": expected_assets,
                    }
                ],
            }
        )
    )


def _zip_bytes() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/workbook.xml", "<workbook/>")
    return buffer.getvalue()


def _successful_fetch(url: str) -> FetchResponse:
    if url == "https://www.nature.com/articles/test-1":
        content = (
            b'<a href="https://media.springernature.com/original/'
            b"springer-static/esm/art%3A10.test%2F1/MediaObjects/supp.pdf"
            b'">PDF</a>'
            b'<a href="https://media.springernature.com/original/'
            b"springer-static/esm/art%3A10.test%2F1/MediaObjects/data.xlsx"
            b'">data</a>'
        )
        content_type = "text/html"
    elif url.endswith(".pdf"):
        content = b"%PDF-1.7\nsynthetic"
        content_type = "application/pdf"
    else:
        content = _zip_bytes()
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return FetchResponse(
        url=url,
        status=200,
        content=content,
        content_type=content_type,
        transport_error="",
    )


def test_config_is_cross_checked_against_source_queue(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    config = tmp_path / "config.json"
    _write_source_queue(queue)
    _write_config(config)

    sources = load_publisher_sources(config, queue)
    assert len(sources) == 1
    assert sources[0].pmc_id == "PMC1"

    _write_config(config, page_url="https://example.com/article")
    with pytest.raises(PublisherSourceError, match="outside the allowlist"):
        load_publisher_sources(config, queue)


def test_publisher_assets_are_discovered_hashed_and_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = tmp_path / "queue.csv"
    config = tmp_path / "config.json"
    _write_source_queue(queue)
    _write_config(config)
    monkeypatch.chdir(tmp_path)

    result = acquire_publisher_sources(
        config,
        queue,
        tmp_path / "cache",
        generated_utc="2026-07-29T00:00:00+00:00",
        workers=1,
        fetch=_successful_fetch,
    )

    assert result["summary"] == {
        "configured_publication_families": 1,
        "source_packages_complete": 1,
        "source_packages_incomplete": 0,
        "expected_assets": 2,
        "validated_assets": 2,
        "complete": True,
        "route_reviews_completed": 0,
    }
    assert result["inputs"][0]["asset"] == "config.json"
    record = result["records"][0]
    assert record["discovery_matches_expected"] is True
    assert record["source_package_status"] == "complete"
    assert {asset["filename"] for asset in record["assets"]} == {
        "data.xlsx",
        "supp.pdf",
    }
    assert all(asset["validated"] for asset in record["assets"])


def test_invalid_or_missing_asset_keeps_package_incomplete(tmp_path: Path) -> None:
    queue = tmp_path / "queue.csv"
    config = tmp_path / "config.json"
    _write_source_queue(queue)
    _write_config(config, expected_assets=1)

    def invalid_fetch(url: str) -> FetchResponse:
        response = _successful_fetch(url)
        if url.endswith(".pdf"):
            return FetchResponse(
                url=url,
                status=200,
                content=b"not a PDF",
                content_type="text/plain",
                transport_error="",
            )
        return response

    result = acquire_publisher_sources(
        config,
        queue,
        tmp_path / "cache",
        generated_utc="2026-07-29T00:00:00+00:00",
        workers=1,
        fetch=invalid_fetch,
    )

    record = result["records"][0]
    assert record["discovery_matches_expected"] is False
    assert record["source_package_status"] == "incomplete"
    assert result["summary"]["complete"] is False
    pdf = next(asset for asset in record["assets"] if asset["filename"] == "supp.pdf")
    assert pdf["validated"] is False
    assert pdf["validation_error"] == "missing PDF signature"


def test_publisher_manifest_write_is_byte_stable(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    result = {
        "schema_version": "test",
        "generated_utc": "2026-07-29T00:00:00+00:00",
    }

    write_publisher_acquisition_result(result, output)
    first = output.read_bytes()
    write_publisher_acquisition_result(result, output)

    assert output.read_bytes() == first
    assert json.loads(first) == result
