from __future__ import annotations

import gzip
import json
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_compose_lipid import release as release

from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import PinError, sha256_file
from forge.corpus.compose_lipid import import_compose_lipid
from forge.corpus.compose_lipid_pretraining import (
    IMPLEMENTATION_SOURCES as PRETRAINING_IMPLEMENTATION_SOURCES,
)
from forge.corpus.compose_lipid_pretraining import (
    audit_split_labels,
    component_labels,
    probe_reconstruction,
    run_pretraining_checks,
    verify_pretraining_checks,
)
from forge.corpus.library_splits import FrozenIdentityFolds
from forge.model.defog_feasibility import AtomState, FeasibilityError
from forge.model.qualified_vocabulary import QualifiedAtomVocabulary
from forge.model.reaction_program_flow import collate_synthesis_program_layouts
from forge.model.sparse_topology_feasibility import tensorize_sparse_row
from forge.model.synthesis_program_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)
from forge.model.synthesis_program_sampling import (
    _atom_capacity_table,
    _terminal_smiles,
    decode_synthesis_program_strict_argmax,
)

REPO = Path(__file__).resolve().parents[1]


def _halogen_case():
    states = (AtomState("C", 0, False), AtomState("Br", 0, False))
    vocabulary = QualifiedAtomVocabulary(states, ("Br",))
    graph = tensorize_sparse_row(
        {"r0_structure_id": "test", "canonical_isomeric_smiles": "CCBr"},
        {s: i for i, s in enumerate(states)},
        preserve_aromaticity=True,
    )
    n = graph.node_count
    record = SynthesisProgramGraphRecord(
        graph=graph,
        canonical_atom_order=np.arange(n),
        program_id="test",
        program_state=1,
        program_depth=1,
        role_states=np.ones(n, dtype=np.int64),
        core_position_states=np.ones(n, dtype=np.int64),
        component_blocks=(SynthesisProgramComponentBlock("role", 1, 0, n),),
        fixed_atom_mask=np.ones(n, dtype=bool),
        fixed_parent_bond_mask=np.array([False, True, True]),
        fixed_closure_bond_mask=np.array([], dtype=bool),
    )
    logits = {
        "nodes": torch.zeros((1, n, 2)),
        "parents": torch.zeros((1, n, n)),
        "parent_bonds": torch.zeros((1, n, 4)),
        "closure_left": torch.zeros((1, 1, n)),
        "closure_right": torch.zeros((1, 1, n)),
        "closure_bonds": torch.zeros((1, 1, 4)),
    }
    return vocabulary, record, logits


def test_bromine_is_opt_in_and_decodes_with_one_bond():
    vocabulary, record, logits = _halogen_case()
    with pytest.raises(FeasibilityError, match="valence policy"):
        _atom_capacity_table(vocabulary.states)
    assert _atom_capacity_table(vocabulary).tolist() == [8, 2]
    layout = collate_synthesis_program_layouts([record], maximum_closures=1)
    decoded, reasons = decode_synthesis_program_strict_argmax(logits, layout, [record], vocabulary)
    assert reasons == (None,)
    assert _terminal_smiles(decoded, 0, 3, 0, vocabulary) == "CCBr"


def test_bromine_double_bond_still_abstains():
    vocabulary, record, logits = _halogen_case()
    bonds = record.graph.parent_bonds.copy()
    bromine = int(np.flatnonzero(record.graph.node_states == 1)[0])
    bonds[bromine] = 1  # sparse double-bond state
    record = replace(record, graph=replace(record.graph, parent_bonds=bonds))
    layout = collate_synthesis_program_layouts([record], maximum_closures=1)
    _, reasons = decode_synthesis_program_strict_argmax(logits, layout, [record], vocabulary)
    assert reasons[0] is not None


def test_existing_valence_capacities_are_bitwise_unchanged():
    from forge.model.vocabulary import load_atom_vocabulary

    old = load_atom_vocabulary(REPO / "results/phase1/product_v3_atom_vocabulary.json")
    assert np.array_equal(
        _atom_capacity_table(old), _atom_capacity_table(QualifiedAtomVocabulary(old))
    )


@pytest.mark.parametrize(
    "state", [AtomState("Br", -1, False), AtomState("Br", 0, True), AtomState("Br", 0, False, 1)]
)
def test_extension_cannot_admit_unqualified_halogen_states(state):
    with pytest.raises(FeasibilityError):
        QualifiedAtomVocabulary((state,), ("Br",))


def test_nonmonovalent_and_missing_extensions_fail():
    with pytest.raises(ValueError, match="exclusively monovalent"):
        QualifiedAtomVocabulary((AtomState("S", 0, False),), ("S",))
    with pytest.raises(ValueError, match="absent"):
        QualifiedAtomVocabulary((AtomState("C", 0, False),), ("Br",))


def _program(variable=()):
    return {
        "source_definition": {
            "roles": {"head": 1},
            "field_to_role": {"head_id": ["head"]},
            "variable": variable,
        },
        "precursor_id_order": [],
    }


def test_labels_are_opaque_and_missing_anchors_are_unresolved():
    labels, complete = component_labels({"primary_metadata": {"head_id": "opaque"}}, _program())
    assert complete and labels == {"head": '{"head_id":"opaque"}'}
    assert component_labels({"primary_metadata": {}}, _program()) == ({}, False)


def test_repeated_program_cannot_pass_a_single_local_cut():
    row = {"primary_metadata": {"head_id": "opaque", "occupancy": 3}}
    result = probe_reconstruction(row, _program(["occupancy"]), object(), {}, None)
    assert result["status"] == "repeated_program_adapter_required"


def test_malformed_heldout_labels_are_counted_without_parsing_graphs(monkeypatch):
    from rdkit import Chem

    def forbidden(*args, **kwargs):
        raise AssertionError("heldout graph access")

    monkeypatch.setattr(Chem, "MolFromSmiles", forbidden)
    db = sqlite3.connect(":memory:")
    db.executescript(
        "CREATE TABLE targets(target_id TEXT,payload TEXT);"
        "CREATE TABLE assignments(target_id TEXT,payload TEXT,forge_split TEXT,provider_split TEXT);"
    )
    source = {
        "target_id": "bad",
        "primary_family": "family",
        "primary_metadata": {"precursor_ids": []},
    }
    assignment = {"combination_signature": "opaque", "test_panels": ["unseen_precursor_identity"]}
    db.execute("INSERT INTO targets VALUES (?,?)", ("bad", json.dumps(source)))
    db.execute(
        "INSERT INTO assignments VALUES (?,?,?,?)",
        ("bad", json.dumps(assignment), "heldout", "test"),
    )
    program = dict(_program(), precursor_id_order=["head"])
    result = audit_split_labels(db, {"family": program})
    assert len(result["malformed_heldout_metadata"]) == 1
    counts = result["populations"]["protected_train"]["unseen_precursor_identity"]
    assert counts["rows"] == 1 and counts["complete_role_labels"] == 0
    assert result["precursor_identity_holdout_qualified"] is False
    db.close()


def test_ambiguous_and_protected_components_abstain():
    import hashlib

    row = {"primary_metadata": {"head_id": "opaque"}, "constitution": "CCC"}
    adapter = SimpleNamespace(
        decompose=lambda *a, **kw: [SimpleNamespace(components=(("head", "CC"),))],
        forward_products=lambda *a, **kw: SimpleNamespace(saturated=False, products=("CCC",)),
    )
    guards = FrozenIdentityFolds()
    # A recovered precursor may be protected even when the product itself is TRAIN.
    identity = hashlib.sha256(b"CC").hexdigest()
    guards.folds[identity].add("heldout")
    result = probe_reconstruction(
        row,
        _program(),
        adapter,
        {"limit": 128, "registry_to_source_roles": {"head": "head"}},
        guards,
    )
    assert result["status"] == "exact_related_transform_protected_precursor"
    assert result["source_program_equivalence_qualified"] is False
    adapter.forward_products = lambda *a, **kw: SimpleNamespace(
        saturated=False, products=("CCC", "CCCC")
    )
    assert (
        probe_reconstruction(row, _program(), adapter, {"limit": 128}, guards)["status"]
        == "ambiguous_forward_site_class"
    )


def test_full_check_retains_large_graphs_and_never_grants_training(release):
    repo, _, _, _, _ = release
    import_compose_lipid(repo, Path("config.json"), Path("import"))
    for name in PRETRAINING_IMPLEMENTATION_SOURCES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    config = {
        "schema_version": "forge.compose_lipid_pretraining_config.v1",
        "import_result": {
            "path": "import/result.json",
            "sha256": str(sha256_file(repo / "import/result.json")),
        },
        "expected_train_rows": 3,
        "neutral_monovalent_extensions": ["Br", "Cl"],
        "related_transforms": {},
    }
    (repo / "pretrain.json").write_text(json.dumps(config))
    result = run_pretraining_checks(repo, Path("pretrain.json"), Path("check"))
    assert result["all_train_graph_roundtrips_and_valences_pass"]
    assert result["source_programs_qualified"] == result["training_calls"] == 0
    vocabulary = json.loads((repo / "check/atom_vocabulary.json").read_text())
    assert vocabulary["maximum_observed"]["atoms"] == 81
    assert {r["symbol"] for r in vocabulary["atom_vocabulary"]} == {
        "C",
        "N",
    }  # held-out O never fits vocabulary
    with gzip.open(repo / "check/train_checks.jsonl.gz", "rt") as stream:
        rows = list(map(json.loads, stream))
    assert len(rows) == 3 and not any(r["training_admitted"] for r in rows)
    assert verify_pretraining_checks(repo, Path("check/result.json")) == result
    with pytest.raises(ComposeLipidError, match="already exists"):
        run_pretraining_checks(repo, Path("pretrain.json"), Path("check"))
    (repo / "check/atom_vocabulary.json").write_text("{}")
    with pytest.raises(PinError, match="changed"):
        verify_pretraining_checks(repo, Path("check/result.json"))


def test_historical_generation_source_authentication_rejects_unknown_bytes(tmp_path):
    from experiments._runtime.historical import HistoricalPinArchiveError
    from experiments.phase1.multireaction.combinatorial_generation_verify import verify

    result = json.loads(
        (REPO / "results/phase1/combinatorial_generation_v1/result.json").read_text()
    )
    result["sources"][0]["sha256"] = "0" * 64
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(result))
    with pytest.raises(HistoricalPinArchiveError, match="no current or archived bytes"):
        verify(REPO, forged)


def test_historical_authentication_cannot_silently_execute_changed_source():
    from experiments.phase1.multireaction.combinatorial_generation_pipeline import (
        contract,
        require_current_execution_sources,
    )

    path = REPO / "configs/multireaction/combinatorial_generation_pipeline_fresh_v2.json"
    config, _ = contract(REPO, path)
    with pytest.raises(PinError, match="pipeline execution source changed"):
        require_current_execution_sources(REPO, config)
