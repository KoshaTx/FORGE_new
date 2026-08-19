from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from forge.bio.oracle_graph import GraphFeatureVocabulary, tensorize_smiles
from forge.bio.oracle_graph_cache import (
    ARCHIVE_FILENAME,
    ARRAY_FIELDS,
    COLLECTIONS,
    METADATA_FILENAME,
    RESULT_FILENAME,
    OracleGraphCacheError,
    build_oracle_graph_cache,
    load_graph_cache,
)

REPO = Path(__file__).resolve().parents[1]


def _write_gzip_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as compressed:
        text = [",".join(fields)]
        text.extend(",".join(row[field] for field in fields) for row in rows)
        compressed.write(("\n".join(text) + "\n").encode())


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    curated = source / "curated.csv.gz"
    oracle_rows = [
        {
            "label": "A1B1C1",
            "model_smiles": "CCN",
            "A_smiles": "CN",
            "B_smiles": "CC=O",
            "C_smiles": "[C-]#[N+]C",
            "expt_Hela": "1.0",
            "expt_Raw": "2.0",
        },
        {
            "label": "A2B2C1",
            "model_smiles": "CCO",
            "A_smiles": "CCN",
            "B_smiles": "CCC=O",
            "C_smiles": "[C-]#[N+]C",
            "expt_Hela": "3.0",
            "expt_Raw": "4.0",
        },
    ]
    _write_gzip_csv(curated, list(oracle_rows[0]), oracle_rows)
    r0 = source / "r0.csv.gz"
    r0_rows = [
        {
            "graph_id": "g1",
            "constitutional_smiles": "CCF",
            "atom_count": "3",
            "undirected_bond_count": "2",
            "directed_edge_count": "4",
        },
        {
            "graph_id": "g2",
            "constitutional_smiles": "C[Si](C)(C)C",
            "atom_count": "5",
            "undirected_bond_count": "4",
            "directed_edge_count": "8",
        },
    ]
    _write_gzip_csv(r0, list(r0_rows[0]), r0_rows)
    virtual = source / "virtual.csv.gz"
    virtual_rows = [
        {
            "source_row_index": "0",
            "source_smiles": "CCS",
            "canonical_isomeric_smiles": "CCS",
        },
        {
            "source_row_index": "1",
            "source_smiles": "CCP",
            "canonical_isomeric_smiles": "CCP",
        },
    ]
    _write_gzip_csv(virtual, list(virtual_rows[0]), virtual_rows)

    vocabulary_smiles = [
        *(
            row[field]
            for row in oracle_rows
            for field in ("model_smiles", "A_smiles", "B_smiles", "C_smiles")
        ),
        *(row["constitutional_smiles"] for row in r0_rows),
        *(row["canonical_isomeric_smiles"] for row in virtual_rows),
    ]
    from rdkit import Chem

    elements: dict[str, int] = {}
    formal_charges: dict[str, int] = {}
    total_degrees: dict[str, int] = {}
    total_hydrogens: dict[str, int] = {}
    hybridizations: dict[str, int] = {}
    bond_types: dict[str, int] = {}
    for smiles in vocabulary_smiles:
        molecule = Chem.MolFromSmiles(smiles)
        for atom in molecule.GetAtoms():
            for container, value in (
                (elements, atom.GetSymbol()),
                (formal_charges, str(atom.GetFormalCharge())),
                (total_degrees, str(atom.GetTotalDegree())),
                (total_hydrogens, str(atom.GetTotalNumHs())),
                (hybridizations, str(atom.GetHybridization())),
            ):
                container[value] = container.get(value, 0) + 1
        for bond in molecule.GetBonds():
            value = str(bond.GetBondType())
            bond_types[value] = bond_types.get(value, 0) + 1
    for required in ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"):
        bond_types.setdefault(required, 0)
    profile = {
        "elements": elements,
        "formal_charges": formal_charges,
        "total_degrees": total_degrees,
        "total_hydrogens": total_hydrogens,
        "hybridizations": hybridizations,
        "bond_types": bond_types,
    }
    graph_corpus = source / "graph_corpus.json"
    graph_corpus.write_text(
        json.dumps(
            {
                "schema_version": "m0_07_oracle_graph_corpus.v2",
                "decision": {"graph_tensorization_authorized": True},
                "representation": {"truncation_allowed": False},
                "pretraining_policy": {
                    "biological_labels_used": False,
                    "virtual_candidates_used": False,
                },
                "graph_profiles": {"r0_pretraining": {"feature_vocabulary": profile}},
            }
        )
    )
    vocabulary = GraphFeatureVocabulary.from_corpus_result(graph_corpus)
    role_graphs = [
        tensorize_smiles(
            row[field],
            vocabulary,
            label=f"{row['label']} {field}",
        )
        for row in oracle_rows
        for field in ("model_smiles", "A_smiles", "B_smiles", "C_smiles")
    ]
    r0_graphs = [
        tensorize_smiles(row["constitutional_smiles"], vocabulary, label=row["graph_id"])
        for row in r0_rows
    ]
    virtual_graphs = [
        tensorize_smiles(
            row["canonical_isomeric_smiles"],
            vocabulary,
            label=row["source_row_index"],
        )
        for row in virtual_rows
    ]
    profile_result = source / "graph_profile.json"
    profile_result.write_text(
        json.dumps(
            {
                "schema_version": "m0_07_oracle_graph_profile.v1",
                "status": "passed_train_only_graph_runtime_gate",
                "decision": {
                    "full_supervised_graph_matrix_authorized": True,
                    "calibration_or_test_labels_used": False,
                },
                "tensorization": {
                    "curated_product_and_roles": {
                        "records": 2,
                        "graphs": 8,
                        "atoms": sum(graph.num_nodes for graph in role_graphs),
                        "directed_edges": sum(graph.num_directed_edges for graph in role_graphs),
                        "maximum_atoms": max(graph.num_nodes for graph in role_graphs),
                    },
                    "r0_pretraining_products": {
                        "records": 2,
                        "graphs": 2,
                        "atoms": sum(graph.num_nodes for graph in r0_graphs),
                        "directed_edges": sum(graph.num_directed_edges for graph in r0_graphs),
                        "maximum_atoms": max(graph.num_nodes for graph in r0_graphs),
                    },
                    "virtual_applicability_products": {
                        "records": 2,
                        "graphs": 2,
                        "atoms": sum(graph.num_nodes for graph in virtual_graphs),
                        "directed_edges": sum(graph.num_directed_edges for graph in virtual_graphs),
                        "maximum_atoms": max(graph.num_nodes for graph in virtual_graphs),
                    },
                },
            }
        )
    )
    tensorization = json.loads(profile_result.read_text())["tensorization"]
    config = source / "config.json"
    inputs = {
        "curated_oracle_data": curated,
        "graph_corpus_result": graph_corpus,
        "graph_profile_result": profile_result,
        "r0_pretraining_ledger": r0,
        "virtual_candidate_library": virtual,
    }
    config.write_text(
        json.dumps(
            {
                "schema_version": "m0_07_oracle_graph_cache_config.v1",
                "generated_utc": "2026-07-30T00:00:00+00:00",
                "seed": 1729,
                "inputs": {
                    name: {
                        "path": str(path.relative_to(tmp_path)),
                        "sha256": _hash(path),
                    }
                    for name, path in inputs.items()
                },
                "expected": {
                    "curated_records": 2,
                    "curated_graphs": tensorization["curated_product_and_roles"]["graphs"],
                    "curated_atoms": tensorization["curated_product_and_roles"]["atoms"],
                    "curated_directed_edges": tensorization["curated_product_and_roles"][
                        "directed_edges"
                    ],
                    "curated_maximum_atoms": tensorization["curated_product_and_roles"][
                        "maximum_atoms"
                    ],
                    "r0_pretraining_records": 2,
                    "r0_pretraining_atoms": tensorization["r0_pretraining_products"]["atoms"],
                    "r0_pretraining_directed_edges": tensorization["r0_pretraining_products"][
                        "directed_edges"
                    ],
                    "r0_pretraining_maximum_atoms": tensorization["r0_pretraining_products"][
                        "maximum_atoms"
                    ],
                    "virtual_records": 2,
                    "virtual_atoms": tensorization["virtual_applicability_products"]["atoms"],
                    "virtual_directed_edges": tensorization["virtual_applicability_products"][
                        "directed_edges"
                    ],
                    "virtual_maximum_atoms": tensorization["virtual_applicability_products"][
                        "maximum_atoms"
                    ],
                },
                "collections": list(COLLECTIONS),
                "policy": {
                    "archive_format": "deterministic_npz_numeric_npy_members",
                    "pickle_allowed": False,
                    "archive_contains_targets": False,
                    "archive_contains_identifiers": False,
                    "metadata_stored_separately": True,
                    "truncation_allowed": False,
                    "source_row_order_preserved": True,
                },
            }
        )
    )
    return config, graph_corpus


def test_cache_is_byte_deterministic_and_roundtrips(tmp_path: Path) -> None:
    config, graph_corpus = _fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_oracle_graph_cache(config, first, tmp_path)
    build_oracle_graph_cache(config, second, tmp_path)

    for filename in (ARCHIVE_FILENAME, METADATA_FILENAME, RESULT_FILENAME):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    cache = load_graph_cache(first / ARCHIVE_FILENAME)
    assert tuple(cache) == COLLECTIONS
    vocabulary = GraphFeatureVocabulary.from_corpus_result(graph_corpus)
    expected = tensorize_smiles("CCN", vocabulary, label="expected")
    observed = cache["oracle_product"].graph(0)
    torch.testing.assert_close(observed.node_features, expected.node_features)
    assert torch.equal(observed.edge_index, expected.edge_index)
    assert torch.equal(observed.edge_features, expected.edge_features)
    assert torch.equal(observed.reverse_edge_index, expected.reverse_edge_index)


def test_archive_is_numeric_only_and_metadata_is_separate(tmp_path: Path) -> None:
    config, _ = _fixture(tmp_path)
    output = tmp_path / "output"
    result = build_oracle_graph_cache(config, output, tmp_path)

    with np.load(output / ARCHIVE_FILENAME, allow_pickle=False) as archive:
        assert len(archive.files) == len(COLLECTIONS) * len(ARRAY_FIELDS)
        assert all(not archive[name].dtype.hasobject for name in archive.files)
        lowered = "|".join(archive.files).lower()
        assert not {"label", "target", "source", "hela", "raw"} & set(lowered.split("|"))
    with gzip.open(output / METADATA_FILENAME, "rt", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    assert metadata[0].keys() == {"collection", "record_index", "record_key"}
    assert {row["collection"] for row in metadata} == {
        "oracle",
        "r0_pretraining",
        "virtual_applicability",
    }
    assert result["policy"]["archive_contains_targets"] is False
    assert result["policy"]["archive_contains_identifiers"] is False
    assert result["decision"]["roundtrip_exact"] is True


def test_cache_rejects_hash_mismatch_and_upstream_truncation(tmp_path: Path) -> None:
    config_path, _ = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["inputs"]["curated_oracle_data"]["sha256"] = "0" * 64
    changed_hash = tmp_path / "bad-hash.json"
    changed_hash.write_text(json.dumps(config))
    with pytest.raises(OracleGraphCacheError, match="hash mismatch"):
        build_oracle_graph_cache(changed_hash, tmp_path / "bad-hash-output", tmp_path)

    config = json.loads(config_path.read_text())
    corpus_path = tmp_path / config["inputs"]["graph_corpus_result"]["path"]
    corpus = json.loads(corpus_path.read_text())
    corpus["representation"]["truncation_allowed"] = True
    corpus_path.write_text(json.dumps(corpus))
    config["inputs"]["graph_corpus_result"]["sha256"] = _hash(corpus_path)
    changed_policy = tmp_path / "bad-policy.json"
    changed_policy.write_text(json.dumps(config))
    with pytest.raises(OracleGraphCacheError, match="truncation"):
        build_oracle_graph_cache(changed_policy, tmp_path / "bad-policy-output", tmp_path)


def test_frozen_cache_artifact_when_present() -> None:
    result_path = REPO / "results/m0_07" / RESULT_FILENAME
    if not result_path.exists():
        pytest.skip("M0-07 graph tensor cache has not been generated")
    result = json.loads(result_path.read_text())
    assert result["schema_version"] == "m0_07_oracle_graph_cache.v1"
    assert result["status"] == "graph_tensor_cache_complete"
    assert result["summary"]["curated_records"] == 1100
    assert result["summary"]["curated_graphs"] == 4400
    assert result["summary"]["r0_pretraining_records"] == 14129
    assert result["summary"]["virtual_records"] == 12276
    assert result["collections"]["r0_pretraining_product"]["maximum_atoms"] == 282
    assert result["decision"]["cache_authorized_for_targets_or_identifiers"] is False
    for filename, metadata in result["artifacts"].items():
        path = REPO / "results/m0_07" / filename
        assert path.stat().st_size == metadata["bytes"]
        assert _hash(path) == metadata["sha256"]
    cache = load_graph_cache(REPO / "results/m0_07" / ARCHIVE_FILENAME)
    assert cache["oracle_product"].graph_count == 1100
    assert cache["r0_pretraining_product"].graph_count == 14129
    assert cache["virtual_applicability_product"].graph_count == 12276
