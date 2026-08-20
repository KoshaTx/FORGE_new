from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from rdkit import Chem

from forge.synthesis.sources.source_ledger import (
    PUBMED_SNAPSHOT_SCHEMA_VERSION,
    SourceLedgerError,
    build_source_ledger,
    normalize_pubmed_esummary,
    sha256_file,
    write_source_ledger,
)


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def _write_raw_lnpdb(path: Path) -> None:
    fields = [
        "LNP_ID",
        "Experiment_ID",
        "IL_name",
        "IL_SMILES",
        "Publication_PMID",
        "Publication_link",
        "IL_head_SMILES",
        "IL_linker_SMILES",
        "IL_tail1_SMILES",
        "IL_tail2_SMILES",
    ]
    rows = [
        {
            "LNP_ID": "LNP_1",
            "Experiment_ID": "EXP_A",
            "IL_name": "lipid-a",
            "IL_SMILES": "CCN",
            "Publication_PMID": "111",
            "Publication_link": "https://doi.org/10.correct/paper",
            "IL_head_SMILES": "CN",
            "IL_linker_SMILES": "",
            "IL_tail1_SMILES": "CCCC",
            "IL_tail2_SMILES": "",
        },
        {
            "LNP_ID": "LNP_2",
            "Experiment_ID": "EXP_B",
            "IL_name": "lipid-b",
            "IL_SMILES": "CCO",
            "Publication_PMID": "333",
            "Publication_link": "https://doi.org/10.normal/paper",
            "IL_head_SMILES": "CN_",
            "IL_linker_SMILES": "COC",
            "IL_tail1_SMILES": "",
            "IL_tail2_SMILES": "CCCCC",
        },
        {
            "LNP_ID": "LNP_3",
            "Experiment_ID": "CATALOG",
            "IL_name": "catalog-lipid",
            "IL_SMILES": "CCC",
            "Publication_PMID": "NA",
            "Publication_link": "https://vendor.example/lipids",
            "IL_head_SMILES": "",
            "IL_linker_SMILES": "COC",
            "IL_tail1_SMILES": "",
            "IL_tail2_SMILES": "",
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_r0(path: Path, source_ids: list[str] | None = None) -> None:
    source_ids = source_ids or ["LNP_1", "LNP_2", "LNP_3"]
    fields = [
        "r0_structure_id",
        "canonical_isomeric_smiles",
        "provenance_json",
    ]
    rows = []
    for index, (smiles, source_id) in enumerate(
        zip(("CCN", "CCO", "CCC"), source_ids, strict=True),
        start=1,
    ):
        rows.append(
            {
                "r0_structure_id": f"R0-{index}",
                "canonical_isomeric_smiles": _canonical(smiles),
                "provenance_json": json.dumps({"lnpdb_v1": {"source_record_ids": [source_id]}}),
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_pubmed_snapshot(path: Path, lnpdb_sha256: str) -> None:
    records = [
        {
            "pmid": "111",
            "doi": "10.wrong/paper",
            "pmc_id": "",
            "title": "Unrelated reported PMID",
            "journal": "Wrong Journal",
            "pubdate": "2018",
        },
        {
            "pmid": "222",
            "doi": "10.correct/paper",
            "pmc_id": "PMC222",
            "title": "Correct lipid paper",
            "journal": "Lipid Journal",
            "pubdate": "2024",
        },
        {
            "pmid": "333",
            "doi": "10.normal/paper",
            "pmc_id": "",
            "title": "Another lipid paper",
            "journal": "Chemistry Journal",
            "pubdate": "2023",
        },
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": PUBMED_SNAPSHOT_SCHEMA_VERSION,
                "retrieved_utc": "2026-07-28T00:00:00+00:00",
                "source": {"provider": "test"},
                "source_sha256": lnpdb_sha256,
                "records": records,
            }
        )
    )


def _write_config(
    path: Path,
    *,
    r0_path: Path,
    lnpdb_path: Path,
    pubmed_path: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "m0_09_lnpdb_route_source_config.v1",
                "role_fields": [
                    "IL_head_SMILES",
                    "IL_linker_SMILES",
                    "IL_tail1_SMILES",
                    "IL_tail2_SMILES",
                ],
                "inputs": {
                    "r0": {
                        "asset": r0_path.name,
                        "expected_sha256": sha256_file(r0_path),
                    },
                    "lnpdb": {
                        "asset": lnpdb_path.name,
                        "expected_sha256": sha256_file(lnpdb_path),
                    },
                    "pubmed_snapshot": {
                        "asset": pubmed_path.name,
                        "expected_sha256": sha256_file(pubmed_path),
                    },
                },
                "publication_identity_overrides": [
                    {
                        "reported_pmid": "111",
                        "publication_link": "https://doi.org/10.correct/paper",
                        "resolved_pmid": "222",
                        "reason": "reported PMID and row DOI disagree",
                    }
                ],
                "non_publication_sources": [
                    {
                        "source_id": "commercial:test",
                        "source_kind": "commercial_catalog",
                        "reported_pmid": "NA",
                        "publication_link": "https://vendor.example/lipids",
                        "title": "Test vendor catalog",
                    }
                ],
                "known_local_sources": [],
            }
        )
    )


def _fixture_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    vendor = tmp_path / "vendor"
    reference = tmp_path / "reference"
    vendor.mkdir()
    reference.mkdir()
    r0 = vendor / "r0.csv"
    lnpdb = vendor / "lnpdb.csv"
    pubmed = reference / "pubmed.json"
    config = tmp_path / "config.json"
    _write_raw_lnpdb(lnpdb)
    _write_r0(r0)
    _write_pubmed_snapshot(pubmed, sha256_file(lnpdb))
    _write_config(config, r0_path=r0, lnpdb_path=lnpdb, pubmed_path=pubmed)
    return config, vendor, reference, r0


def test_builds_row_level_source_queue_without_inventing_route_closure(
    tmp_path: Path,
) -> None:
    config, vendor, reference, _ = _fixture_paths(tmp_path)

    result, sources, components = build_source_ledger(
        config,
        vendor,
        reference,
        generated_utc="2026-07-28T00:00:00+00:00",
    )

    assert result["summary"]["lnpdb_records"] == 3
    assert result["summary"]["lnpdb_unique_lipids"] == 3
    assert result["summary"]["raw_lnpdb_to_r0_exact_match"] is True
    assert result["summary"]["unique_publications"] == 2
    assert result["summary"]["commercial_sources"] == 1
    assert result["summary"]["publication_identity_corrections"] == 1
    assert result["summary"]["normalized_role_components"] == 5
    assert result["summary"]["role_components"]["head"]["invalid_components"] == 1

    corrected = next(source for source in sources if source["source_id"] == "222")
    assert corrected["reported_pmids_json"] == '["111"]'
    assert corrected["title"] == "Correct lipid paper"
    assert corrected["pmc_id"] == "PMC222"
    assert json.loads(corrected["source_identity_qa_json"])

    commercial = next(source for source in sources if source["source_id"] == "commercial:test")
    assert commercial["acquisition_status"] == "live_source_requires_snapshot"
    assert commercial["source_kind"] == "commercial_catalog"

    assert all(component["route_evidence_status"] == "not_assessed" for component in components)
    assert all(
        component["procurement_evidence_status"] == "not_assessed" for component in components
    )
    assert all(component["execution_closure_status"] == "unknown" for component in components)

    output = tmp_path / "result"
    write_source_ledger(result, sources, components, output)
    assert (output / "route_source_ledger.json").exists()
    assert sha256_file(output / "source_queue.csv") == result["outputs"]["source_queue"]["sha256"]
    assert (
        sha256_file(output / "component_source_ledger.csv")
        == result["outputs"]["component_source_ledger"]["sha256"]
    )


def test_fails_when_raw_lnpdb_does_not_match_r0_provenance(tmp_path: Path) -> None:
    config, vendor, reference, r0 = _fixture_paths(tmp_path)
    _write_r0(r0, source_ids=["LNP_1", "LNP_2", "NOT_LNP_3"])
    _write_config(
        config,
        r0_path=r0,
        lnpdb_path=vendor / "lnpdb.csv",
        pubmed_path=reference / "pubmed.json",
    )

    with pytest.raises(SourceLedgerError, match="do not exactly match R0 provenance"):
        build_source_ledger(config, vendor, reference)


def test_normalize_pubmed_esummary_requires_exact_requested_pmids() -> None:
    payload = {
        "result": {
            "uids": ["2"],
            "2": {
                "title": "A lipid paper",
                "fulljournalname": "Journal",
                "pubdate": "2024",
                "articleids": [
                    {"idtype": "doi", "value": "10.example/test"},
                    {"idtype": "pmc", "value": "PMC2"},
                ],
            },
        }
    }

    normalized = normalize_pubmed_esummary(
        payload,
        ["2"],
        retrieved_utc="2026-07-28T00:00:00+00:00",
        source_sha256="abc",
    )
    assert normalized["records"][0]["doi"] == "10.example/test"

    with pytest.raises(SourceLedgerError, match="PMID mismatch"):
        normalize_pubmed_esummary(
            payload,
            ["1", "2"],
            retrieved_utc="2026-07-28T00:00:00+00:00",
            source_sha256="abc",
        )
