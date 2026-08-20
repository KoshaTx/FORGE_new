from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

from forge.route.sources.source_ledger import SOURCE_COLUMNS
from forge.route.sources.source_priority import (
    SourcePriorityError,
    build_source_priority,
    write_source_priority,
)
from forge.route.sources.supervision_inventory import sha256_file

REPO = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, fieldnames: list[str] | tuple[str, ...], rows: list[dict]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _source_row(
    rank: int,
    source_id: str,
    *,
    pmid: str,
    link: str,
    source_kind: str = "publication",
    acquisition_status: str = "pmc_identifier_available",
    review_bucket: str = "pmc_first_pass",
) -> dict[str, str | int]:
    row: dict[str, str | int] = {column: "" for column in SOURCE_COLUMNS}
    row.update(
        {
            "review_rank": rank,
            "review_bucket": review_bucket,
            "source_id": source_id,
            "source_kind": source_kind,
            "pmid": pmid,
            "title": f"Source {source_id}",
            "acquisition_status": acquisition_status,
            "route_review_status": "not_reviewed",
            "lnpdb_record_count": 1,
            "unique_lipid_count": 1,
            "experiment_count": 1,
            "il_name_count": 1,
            "unique_component_count": 1,
            "unique_tail1_count": 1,
            "experiment_ids_json": json.dumps([f"EXP_{source_id}"]),
            "reported_pmids_json": json.dumps([pmid or "NA"]),
            "publication_links_json": json.dumps([link]),
            "source_identity_qa_json": "[]",
            "local_source_assets_json": "[]",
        }
    )
    return row


def _lnpdb_row(
    pmid: str,
    link: str,
    *,
    model: str,
    value: str,
) -> dict[str, str]:
    return {
        "Publication_PMID": pmid,
        "Publication_link": link,
        "Model": model,
        "Model_type": "Mouse" if model == "in_vivo" else "HeLa",
        "Model_target": "liver" if model == "in_vivo" else "in_vitro",
        "Route_of_administration": "intravenous" if model == "in_vivo" else "in_vitro",
        "Cargo": "mRNA",
        "Cargo_type": "reporter",
        "Experiment_method": "luminescence_normalized",
        "Experiment_value": value,
    }


def _fixture(tmp_path: Path) -> Path:
    source_queue = tmp_path / "source_queue.csv"
    component_ledger = tmp_path / "component_source_ledger.csv"
    lnpdb = tmp_path / "lnpdb.csv"
    agile = tmp_path / "agile_components.csv.gz"
    reviews = tmp_path / "paper_reviews.json"
    motif_result = tmp_path / "motif_result.json"
    motif_ledger = tmp_path / "motif_ledger.csv.gz"

    source_rows = [
        _source_row(1, "111", pmid="111", link="https://doi.org/111"),
        _source_row(2, "222", pmid="222", link="https://doi.org/222"),
        _source_row(
            3,
            "catalog:test",
            pmid="",
            link="https://vendor.example",
            source_kind="commercial_catalog",
            acquisition_status="live_source_requires_snapshot",
            review_bucket="commercial_procurement_review",
        ),
    ]
    _write_csv(source_queue, SOURCE_COLUMNS, source_rows)

    component_fields = [
        "role",
        "parse_status",
        "canonical_smiles",
        "source_count",
        "source_ids_json",
    ]
    _write_csv(
        component_ledger,
        component_fields,
        [
            {
                "role": "tail1",
                "parse_status": "parsed",
                "canonical_smiles": "CCCC",
                "source_count": 1,
                "source_ids_json": '["111"]',
            },
            {
                "role": "tail1",
                "parse_status": "parsed",
                "canonical_smiles": "CCCCC",
                "source_count": 1,
                "source_ids_json": '["222"]',
            },
            {
                "role": "linker",
                "parse_status": "parsed",
                "canonical_smiles": "COC",
                "source_count": 2,
                "source_ids_json": '["111", "222"]',
            },
        ],
    )
    _write_csv(agile, ["canonical_smiles"], [{"canonical_smiles": "CCCC"}])
    _write_csv(
        lnpdb,
        [
            "Publication_PMID",
            "Publication_link",
            "Model",
            "Model_type",
            "Model_target",
            "Route_of_administration",
            "Cargo",
            "Cargo_type",
            "Experiment_method",
            "Experiment_value",
        ],
        [
            _lnpdb_row("111", "https://doi.org/111", model="in_vitro", value="1.0"),
            _lnpdb_row("222", "https://doi.org/222", model="in_vivo", value="2.0"),
            _lnpdb_row(
                "NA",
                "https://vendor.example",
                model="",
                value="NA",
            ),
        ],
    )
    reviews.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "pmid": "111",
                        "review_id": "review_111",
                        "source_asset": {"review_status": "chemistry_pages_visually_reviewed"},
                        "l1_assembly_families": [
                            {
                                "reactant_roles": [
                                    "amine head",
                                    "aldehyde tail",
                                    "isocyanide tail",
                                ]
                            }
                        ],
                        "l2_route_families": [],
                        "review_conclusion": {"upstream_component_routes": 1},
                    }
                ]
            }
        )
    )
    motif_result.write_text(json.dumps({"sources": {"transfer_222": {"pmid": "222"}}}))
    _write_csv(
        motif_ledger,
        [
            "record_id",
            "source_id",
            "motif_classes_json",
            "source_attachment_mapping_status",
            "proposed_ugi_component_smiles",
            "frozen_ugi_forward_products",
            "computationally_route_complete",
            "biological_label_inherited",
        ],
        [
            {
                "record_id": "motif-222",
                "source_id": "transfer_222",
                "motif_classes_json": '["branched", "unsaturated"]',
                "source_attachment_mapping_status": "exact_mapped",
                "proposed_ugi_component_smiles": "CCCCC=O",
                "frozen_ugi_forward_products": 1,
                "computationally_route_complete": "true",
                "biological_label_inherited": "false",
            }
        ],
    )

    inputs = {
        "source_queue": source_queue,
        "component_source_ledger": component_ledger,
        "lnpdb": lnpdb,
        "agile_component_ledger": agile,
        "paper_route_reviews": reviews,
        "motif_transfer_result": motif_result,
        "motif_transfer_ledger": motif_ledger,
    }
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "m0_09_source_priority_config.v1",
                "generated_utc": "2026-07-30T23:45:00Z",
                "seed": 0,
                "inputs": {
                    label: {
                        "path": str(path.relative_to(tmp_path)),
                        "expected_sha256": sha256_file(path),
                    }
                    for label, path in inputs.items()
                },
                "policy": {
                    "hydrophobic_roles": ["linker", "tail1", "tail2"],
                    "axis_weights": {
                        "motif_distinctiveness": 1,
                        "biological_provenance": 1,
                        "agile_exact_novelty": 1,
                        "method_accessibility": 1,
                        "ugi_handle_conversion": 1,
                        "exact_component_redundancy": 1,
                    },
                    "missing_evidence_score": 0,
                    "thresholds": {
                        "motif_high_class_count": 2,
                        "novelty_high_fraction": 0.75,
                        "redundancy_low_shared_fraction": 0.25,
                        "redundancy_mid_shared_fraction": 0.75,
                    },
                },
            }
        )
    )
    return config


def test_priority_overlay_preserves_acquisition_rows_and_keeps_missing_explicit(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)
    result, rows = build_source_priority(config, tmp_path)

    assert result["summary"]["source_records"] == 3
    assert result["summary"]["acquisition_publications_preserved"] == 2
    assert result["summary"]["acquisition_commercial_sources_preserved"] == 1
    assert result["summary"]["chemistry_review_complete_sources"] == 1
    assert result["summary"]["active_next_review_sources"] == 2

    by_id = {row["source_id"]: row for row in rows}
    reviewed = by_id["111"]
    assert reviewed["review_rank"] == "1"
    assert reviewed["next_review_rank"] == ""
    assert reviewed["review_completion_state"] == "chemistry_review_complete"
    assert (
        reviewed["ugi_handle_conversion_state"]
        == "native_ugi_l1_with_reviewed_upstream_component_routes"
    )
    assert json.loads(reviewed["missing_priority_axes_json"]) == ["motif_distinctiveness"]

    transfer = by_id["222"]
    assert transfer["next_review_rank"] == 1
    assert transfer["motif_distinctiveness_score"] == 2
    assert transfer["biological_provenance_score"] == 2
    assert transfer["agile_exact_novelty_score"] == 2
    assert transfer["method_accessibility_score"] == 3
    assert transfer["ugi_handle_conversion_score"] == 3
    assert transfer["exact_component_redundancy_score"] == 1
    assert json.loads(transfer["missing_priority_axes_json"]) == []

    missing = by_id["catalog:test"]
    assert missing["priority_score"] == 0
    assert missing["method_accessibility_score"] == 0
    assert set(json.loads(missing["missing_priority_axes_json"])) == {
        "motif_distinctiveness",
        "ugi_handle_conversion",
    }


def test_priority_output_is_deterministic_and_hash_pinned(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    first_result, first_rows = build_source_priority(config, tmp_path)
    second_result, second_rows = build_source_priority(config, tmp_path)
    assert first_result == second_result
    assert first_rows == second_rows

    first = tmp_path / "first"
    second = tmp_path / "second"
    write_source_priority(first_result, first_rows, first)
    write_source_priority(second_result, second_rows, second)
    for filename in ("source_priority_result.json", "source_priority_queue.csv"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert (
        sha256_file(first / "source_priority_queue.csv")
        == first_result["artifacts"]["source_priority_queue.csv"]["sha256"]
    )


def test_priority_rejects_biological_label_inheritance(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    payload = json.loads(config.read_text())
    motif_path = tmp_path / payload["inputs"]["motif_transfer_ledger"]["path"]
    with gzip.open(motif_path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["biological_label_inherited"] = "true"
    _write_csv(motif_path, fieldnames, rows)
    payload["inputs"]["motif_transfer_ledger"]["expected_sha256"] = sha256_file(motif_path)
    config.write_text(json.dumps(payload))

    with pytest.raises(SourcePriorityError, match="improperly inherits a biological label"):
        build_source_priority(config, tmp_path)


def test_priority_rejects_stale_input_hash(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    payload = json.loads(config.read_text())
    payload["inputs"]["source_queue"]["expected_sha256"] = "0" * 64
    config.write_text(json.dumps(payload))
    with pytest.raises(SourcePriorityError, match="hash mismatch for priority input source_queue"):
        build_source_priority(config, tmp_path)


@pytest.mark.needs_vendor
def test_frozen_source_priority_artifacts_reproduce_exactly() -> None:
    config = REPO / "configs/route/m0_09_source_priority.json"
    result_path = REPO / "results/m0_09/source_priority_result.json"
    queue_path = REPO / "results/m0_09/source_priority_queue.csv"
    if not (REPO / "data/vendor/lnpdb_fc7c389.csv").exists():
        pytest.skip("run `make vendor` first")

    rebuilt, rows = build_source_priority(config, REPO)
    assert rebuilt == json.loads(result_path.read_text())
    assert rebuilt["summary"]["source_records"] == 43
    assert rebuilt["summary"]["acquisition_publications_preserved"] == 42
    assert rebuilt["summary"]["acquisition_commercial_sources_preserved"] == 1
    assert rebuilt["summary"]["chemistry_review_complete_sources"] == 4
    assert rebuilt["summary"]["active_next_review_sources"] == 39
    assert sha256_file(queue_path) == rebuilt["artifacts"]["source_priority_queue.csv"]["sha256"]

    original_by_id: dict[str, dict[str, str]] = {}
    with (REPO / "results/m0_09/source_queue.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            original_by_id[row["source_id"]] = row
    for row in rows:
        assert {column: row[column] for column in SOURCE_COLUMNS} == original_by_id[
            row["source_id"]
        ]
