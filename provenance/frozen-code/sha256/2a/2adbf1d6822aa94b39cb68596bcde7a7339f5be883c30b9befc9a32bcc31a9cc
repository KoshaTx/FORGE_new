from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_compose_lipid import release as release

from forge.assembly.component_constraints import ComponentConstraint, propagate_component_labels
from forge.assembly.compose_lipid import ComposeLipidError
from forge.core.hashing import PinError, sha256_file
from forge.corpus.compose_lipid import import_compose_lipid
from forge.corpus.compose_lipid_components import (
    CONFIG_SCHEMA,
    IMPLEMENTATION,
    _read_rows,
    classify_components,
    enumerate_candidates,
    run_component_constraints,
    verify_component_constraints,
)
from forge.corpus.compose_lipid_pretraining import run_pretraining_checks
from forge.corpus.library_splits import FrozenIdentityFolds

REPO = Path(__file__).resolve().parents[1]


def _constraint(target, labels, candidates, family="family", role="head"):
    return ComponentConstraint(
        target, tuple((family, role, label) for label in labels), tuple(candidates)
    )


def test_shared_label_propagates_through_more_than_one_product():
    rows = [
        _constraint("first", ["a", "b"], [("CC", "N"), ("CCC", "O")]),
        _constraint("second", ["b", "c"], [("N", "CO"), ("O", "CN")]),
        _constraint("anchor", ["c"], [("CN",)]),
    ]
    result = propagate_component_labels(rows)
    assert result.surviving_candidates["first"] == (("CCC", "O"),)
    assert result.singleton_component_witness_targets == ("anchor", "first", "second")
    assert result == propagate_component_labels(list(reversed(rows)))


def test_family_and_role_scope_prevent_label_collisions():
    rows = [
        _constraint("first", ["same"], [("CC",)]),
        _constraint("other_family", ["same"], [("CCC",)], family="different"),
        _constraint("other_role", ["same"], [("N",)], role="tail"),
    ]
    assert not propagate_component_labels(rows).conflicting_targets


def test_conflict_quarantines_entire_connected_component_without_arbitrary_winner():
    rows = [
        _constraint("first", ["a"], [("CC",)]),
        _constraint("second", ["a", "b"], [("CCC", "N")]),
        _constraint("third", ["b"], [("N",)]),
        _constraint("independent", ["c"], [("O",)]),
    ]
    result = propagate_component_labels(rows)
    assert result.conflicting_targets == ("first", "second", "third")
    assert result.singleton_component_witness_targets == ("independent",)


def test_arc_consistent_unsatisfiable_cycle_is_never_called_a_global_witness():
    # a=b, b=c, a!=c: nonempty pairwise supports do not prove joint consistency.
    result = propagate_component_labels(
        [
            _constraint("ab", ["a", "b"], [("C", "C"), ("N", "N")]),
            _constraint("bc", ["b", "c"], [("C", "C"), ("N", "N")]),
            _constraint("ac", ["a", "c"], [("C", "N"), ("N", "C")]),
        ]
    )
    assert not result.singleton_component_witness_targets
    assert all(len(values) == 2 for values in result.surviving_candidates.values())


def test_duplicate_target_or_malformed_candidate_fails_loudly():
    row = _constraint("duplicate", ["a"], [("C",)])
    with pytest.raises(ValueError, match="duplicate target"):
        propagate_component_labels([row, row])
    with pytest.raises(ValueError, match="arity"):
        _constraint("bad", ["a", "b"], [("C",)])


def _record(target, candidates, *, label="same", original="ambiguous_related_transform"):
    return {
        "target_id": target,
        "family": "family",
        "labels": {"head": label},
        "original_status": original,
        "candidates": [
            {"components": {"head": s}, "forward_unique": unique} for s, unique in candidates
        ],
    }


def test_newly_resolved_protected_precursor_is_quarantined_and_all_weights_stay_zero():
    guards = FrozenIdentityFolds()
    guards.folds[hashlib.sha256(b"CC").hexdigest()].add("heldout")
    records = [
        _record("ambiguous", [("CC", True), ("CCC", True)]),
        _record("anchor", [("CC", True)], original="exact_related_transform_protected_precursor"),
    ]
    view, counts = classify_components(records, guards)
    assert counts["family"]["new_singleton_witnesses"] == 1
    assert all(r["disposition"] == "quarantine_protected_precursor" for r in view)
    assert all(r["training_admitted"] is False and r["training_weight"] == 0 for r in view)


def test_label_consistency_cannot_resolve_forward_site_ambiguity():
    rows = [_record("first", [("CC", False)]), _record("second", [("CC", True)])]
    view, _ = classify_components(rows, FrozenIdentityFolds())
    assert view[0]["reason"] == "ambiguous_forward_site_class"
    assert view[1]["reason"] == "source_program_qualification_missing"
    assert not any(r["training_admitted"] for r in view)


def test_known_precursor_exclusion_survives_later_label_conflict():
    rows = [
        _record(
            "protected", [("CC", True)], original="exact_related_transform_protected_precursor"
        ),
        _record("conflicting", [("CCC", True)]),
    ]
    view, _ = classify_components(rows, FrozenIdentityFolds())
    assert view[0]["reason"] == "source_label_structure_conflict"
    assert view[0]["disposition"] == "quarantine_protected_precursor"


def _candidate_inputs():
    row = {"primary_metadata": {"head_id": "label"}, "constitution": "CCN"}
    program = {
        "source_definition": {"roles": {"head": 1}, "field_to_role": {"head_id": ["head"]}},
        "precursor_id_order": [],
    }
    binding = {"limit": 128, "registry_to_source_roles": {"head": "head"}}
    adapter = SimpleNamespace(
        decompose=lambda *args, **kw: [SimpleNamespace(components=(("head", "CC"),))],
        forward_products=lambda *args, **kw: SimpleNamespace(
            saturated=False, products=("CCN", "CN")
        ),
    )
    return row, program, binding, adapter


def test_forward_ambiguous_candidate_is_retained_in_inverse_domain():
    result = enumerate_candidates(*_candidate_inputs())
    assert result["candidates"] == [{"components": {"head": "CC"}, "forward_unique": False}]


def test_saturated_search_never_returns_partial_candidate_set():
    row, program, binding, adapter = _candidate_inputs()
    adapter.forward_products = lambda *args, **kw: SimpleNamespace(
        saturated=True, products=("CCN",)
    )
    assert enumerate_candidates(row, program, binding, adapter)["candidates"] == []


def test_duplicate_source_label_candidates_do_not_acquire_arbitrary_order():
    rows = [_record("still_ambiguous", [("C", True), ("N", True)])]
    view, _ = classify_components(rows, FrozenIdentityFolds())
    assert view[0]["remaining_candidates"] == 2
    assert view[0]["components"] == []
    assert view[0]["reason"] == "component_assignment_unresolved"


def test_full_pipeline_replay_and_tamper_detection_without_heldout_graph_parsing(
    release, monkeypatch
):
    repo, _, _, _, _ = release
    import_compose_lipid(repo, Path("config.json"), Path("import"))
    for name in (
        *IMPLEMENTATION,
        "forge/corpus/compose_lipid_pretraining.py",
        "forge/model/qualified_vocabulary.py",
        "forge/model/synthesis_program_sampling.py",
        "forge/assembly/registry.py",
        "forge/assembly/program.py",
        "forge/chemistry/reactive_sites.py",
        "forge/model/defog_feasibility.py",
    ):
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
        "neutral_monovalent_extensions": [],
        "related_transforms": {},
    }
    (repo / "pretrain.json").write_text(json.dumps(config))
    run_pretraining_checks(repo, Path("pretrain.json"), Path("pretrain"))
    config = {
        "schema_version": CONFIG_SCHEMA,
        "pretraining_audit": {
            "path": "pretrain/result.json",
            "sha256": str(sha256_file(repo / "pretrain/result.json")),
        },
        "policy": {
            "fit_split": "train",
            "label_scope": "family_and_role",
            "training_calls": 0,
            "generation_calls": 0,
            "heldout_structure_access": False,
            "seed": 0,
        },
    }
    (repo / "components.json").write_text(json.dumps(config))
    # The O-containing molecules in this fixture occur only outside TRAIN. Historical guards
    # contain CCC and may be normalized for exclusion, but no O graph may be interpreted here.
    from rdkit import Chem

    original = Chem.MolFromSmiles

    def guard(smiles, *args, **kwargs):
        assert "O" not in smiles, "held-out structure parsed"
        return original(smiles, *args, **kwargs)

    monkeypatch.setattr(Chem, "MolFromSmiles", guard)
    result = run_component_constraints(repo, Path("components.json"), Path("output"))
    assert result["train_rows"] == 3
    assert result["training_admitted_rows"] == 0
    assert verify_component_constraints(repo, Path("output/result.json")) == result
    replay = run_component_constraints(repo, Path("components.json"), Path("replay"))
    assert result["by_family"] == replay["by_family"]
    for name in result["artifacts"]:
        assert result["artifacts"][name]["sha256"] == replay["artifacts"][name]["sha256"]
    assert all(
        r["disposition"] == "hold_unqualified"
        for r in _read_rows(repo / "output/admission_view.jsonl.gz")
    )
    with pytest.raises(ComposeLipidError, match="already exists"):
        run_component_constraints(repo, Path("components.json"), Path("output"))
    result["training_admitted_rows"] = 1
    (repo / "output/result.json").write_text(json.dumps(result))
    with pytest.raises(ComposeLipidError, match="cannot claim"):
        verify_component_constraints(repo, Path("output/result.json"))
    (repo / "output/admission_view.jsonl.gz").write_bytes(b"tampered")
    with pytest.raises(PinError, match="changed"):
        verify_component_constraints(repo, Path("output/result.json"))
