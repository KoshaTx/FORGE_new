from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from experiments.archive.phase1.synthesis_audits.source_review import (
    SourceReviewError,
    build_source_review_index,
    write_source_review_index,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


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


def _write_manifests(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    queue = tmp_path / "queue.csv"
    components = tmp_path / "components.csv"
    pmc_manifest = tmp_path / "pmc.json"
    publisher_manifest = tmp_path / "publisher.json"
    cache = tmp_path / "cache"
    _write_source_queue(queue)
    with components.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["component_id", "role", "source_pmids_json"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "component_id": "head-1",
                    "role": "head",
                    "source_pmids_json": '["1001"]',
                },
                {
                    "component_id": "tail-1",
                    "role": "tail1",
                    "source_pmids_json": '["1003"]',
                },
                {
                    "component_id": "tail-2",
                    "role": "tail1",
                    "source_pmids_json": '["9999"]',
                },
            ]
        )

    records = []
    for index in range(1, 27):
        pmc_id = f"PMC{index}"
        main_content = f"main text {index}".encode()
        main_relative = Path(pmc_id) / "bioc.xml"
        main_path = cache / "m0_09_pmc" / main_relative
        main_path.parent.mkdir(parents=True, exist_ok=True)
        main_path.write_bytes(main_content)
        record = {
            "pmc_id": pmc_id,
            "supplement_status": "publisher_discovery_required",
            "jats_supplement_links": [],
            "endpoints": {
                "bioc_full_text": {
                    "cache_asset": str(main_relative),
                    "sha256": _sha256(main_content),
                    "bytes": len(main_content),
                    "url": f"https://example.test/{pmc_id}/main",
                },
                "europe_pmc_supplements": {
                    "cache_asset": "",
                    "sha256": "",
                },
            },
            "supplement_archive": {"status": "not_acquired"},
        }
        records.append(record)

    official_content = b"%PDF-1.7\nofficial supplement"
    official_path = cache / "m0_09_pmc" / "PMC1" / "supp.zip"
    with zipfile.ZipFile(official_path, "w") as archive:
        archive.writestr("supp.pdf", official_content)
    official_bytes = official_path.read_bytes()
    records[0]["supplement_status"] = "official_archive_acquired"
    records[0]["jats_supplement_links"] = ["supp.pdf"]
    records[0]["endpoints"]["europe_pmc_supplements"] = {
        "cache_asset": "PMC1/supp.zip",
        "sha256": _sha256(official_bytes),
    }
    records[0]["supplement_archive"] = {"status": "validated_archive_acquired"}
    pmc_manifest.write_text(json.dumps({"records": records}))

    publisher_content = b"%PDF-1.7\npublisher supplement"
    publisher_relative = Path("PMC2") / "assets" / "publisher.pdf"
    publisher_path = cache / "m0_09_publishers" / publisher_relative
    publisher_path.parent.mkdir(parents=True, exist_ok=True)
    publisher_path.write_bytes(publisher_content)
    publisher_manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "pmc_id": "PMC2",
                        "source_package_status": "complete",
                        "assets": [
                            {
                                "filename": "publisher.pdf",
                                "cache_asset": str(publisher_relative),
                                "bytes": len(publisher_content),
                                "sha256": _sha256(publisher_content),
                                "url": "https://example.test/publisher.pdf",
                                "validated": True,
                            }
                        ],
                    }
                ]
            }
        )
    )
    return queue, components, pmc_manifest, publisher_manifest, cache


def test_review_index_merges_and_verifies_both_source_channels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue, components, pmc, publisher, cache = _write_manifests(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = build_source_review_index(
        queue,
        components,
        pmc,
        publisher,
        cache,
        generated_utc="2026-07-29T00:00:00+00:00",
    )

    assert result["summary"] == {
        "pmc_publication_families": 26,
        "source_packages_complete": 2,
        "source_packages_supplement_unavailable": 24,
        "machine_readable_main_texts": 26,
        "supplementary_review_assets": 2,
        "route_reviews_completed": 0,
    }
    assert result["inputs"][0]["asset"] == "queue.csv"
    assert result["component_source_package_coverage"] == {
        "total_components": 3,
        "any_complete_source_package": 1,
        "only_incomplete_pmc_source_package": 1,
        "non_pmc_or_unlinked_source_package": 1,
        "by_role": {
            "head": {
                "any_complete_source_package": 1,
                "only_incomplete_pmc_source_package": 0,
                "non_pmc_or_unlinked_source_package": 0,
            },
            "tail1": {
                "any_complete_source_package": 0,
                "only_incomplete_pmc_source_package": 1,
                "non_pmc_or_unlinked_source_package": 1,
            },
        },
    }
    records = {record["pmc_id"]: record for record in result["records"]}
    assert records["PMC1"]["source_package_channel"] == ("europe_pmc_supplement_archive")
    assert records["PMC2"]["source_package_channel"] == "publisher_page"
    extracted = cache / records["PMC1"]["supplements"][0]["cache_asset"]
    assert extracted.read_bytes() == b"%PDF-1.7\nofficial supplement"
    assert records["PMC3"]["gap_reason"] == "publisher_discovery_required"


def test_review_index_rejects_hash_mismatch(tmp_path: Path) -> None:
    queue, components, pmc, publisher, cache = _write_manifests(tmp_path)
    (cache / "m0_09_pmc/PMC3/bioc.xml").write_bytes(b"tampered")

    with pytest.raises(SourceReviewError, match="hash mismatch"):
        build_source_review_index(queue, components, pmc, publisher, cache)


def test_review_index_rejects_unsafe_archive_member(tmp_path: Path) -> None:
    queue, components, pmc, publisher, cache = _write_manifests(tmp_path)
    payload = json.loads(pmc.read_text())
    payload["records"][0]["jats_supplement_links"] = ["../supp.pdf"]
    pmc.write_text(json.dumps(payload))

    with pytest.raises(SourceReviewError, match="unsafe supplementary"):
        build_source_review_index(queue, components, pmc, publisher, cache)


def test_source_review_index_write_is_byte_stable(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    result = {
        "schema_version": "test",
        "generated_utc": "2026-07-29T00:00:00+00:00",
    }

    write_source_review_index(result, output)
    first = output.read_bytes()
    write_source_review_index(result, output)

    assert output.read_bytes() == first
    assert json.loads(first) == result
