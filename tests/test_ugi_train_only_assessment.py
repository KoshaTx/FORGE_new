"""Train-only masking, constitutional identities and unchanged attempt denominators."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import PinError, sha256_file
from forge.core.io import csv_gz_bytes
from forge.model.common_ugi_benchmark import CommonUgiAttempt, CommonUgiBenchmarkError
from forge.model.ugi_train_only_assessment import (
    UgiTrainOnlyAssessmentError,
    assess_ugi_train_only_attempts,
    load_ugi_training_identities,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def adapter() -> Ugi3AssemblyAdapter:
    return Ugi3AssemblyAdapter.from_registry(REPO / "data/vendor/qualified_reactions_v1.json")


@pytest.fixture
def example_rows(adapter: Ugi3AssemblyAdapter) -> list[dict[str, str]]:
    registry = json.loads(adapter.registry_path.read_text())
    reaction = next(
        row for row in registry["reactions"] if row["reaction_id"] == adapter.reaction_id
    )
    rows = []
    for example in reaction["known_positive_examples"]:
        row = {"primary_product_fold": "train", "canonical_product_smiles": example["expected"]}
        for role, smiles in zip(adapter.roles, example["reactants"], strict=True):
            row[f"{role}_smiles"] = smiles
            row[f"{role}_family_fold"] = "train"
        rows.append(row)
    return rows


def _write_assignments(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "assignments.csv.gz"
    path.write_bytes(csv_gz_bytes(rows, list(rows[0])))
    return path


def _attempt(index: int, smiles: str | None, *, status: str = "generated") -> CommonUgiAttempt:
    return CommonUgiAttempt.from_mapping(
        {
            "method_id": "test",
            "seed": 7,
            "attempt_index": index,
            "status": status,
            "product_smiles": smiles,
            "method_visible_component_ids": [],
            "generator_calls": 1,
            "reaction_calls": 0,
            "route_calls": 0,
            "oracle_calls": 0,
            "wall_seconds": 0.0,
        }
    )


def test_nontrain_malformed_structures_are_never_parsed(tmp_path, adapter, example_rows) -> None:
    train = example_rows[0]
    nontrain = {
        key: "unparseable sealed molecular field" if "smiles" in key else "heldout" for key in train
    }
    path = _write_assignments(
        tmp_path, [nontrain, train, dict(nontrain, primary_product_fold="cal")]
    )
    identities = load_ugi_training_identities(path, adapter, sha256_file(path))
    assert identities.provenance["train_rows"] == 1
    assert identities.provenance["rows_seen"] == 3
    assert identities.provenance["non_train_rows_masked_before_structure_access"] == 2
    assert identities.provenance["heldout_structures_interpreted"] is False
    assert identities.provenance["unique_training_products"] == 1
    assert identities.provenance["unique_training_components_by_role"] == {
        role: 1 for role in adapter.roles
    }
    assert identities.provenance["assignments"]["sha256"] == sha256_file(path)


def test_nontrain_mask_precedes_every_other_field_access(
    tmp_path, adapter, example_rows, monkeypatch
) -> None:
    class SealedRow(dict):
        def get(self, key, default=None):
            assert key == "primary_product_fold", f"sealed field accessed: {key}"
            return super().get(key, default)

        def __getitem__(self, key):
            assert key == "primary_product_fold", f"sealed field accessed: {key}"
            return super().__getitem__(key)

    path = _write_assignments(tmp_path, [example_rows[0]])
    monkeypatch.setattr(
        "forge.model.ugi_train_only_assessment.iter_csv",
        lambda _: iter([SealedRow(primary_product_fold="heldout"), example_rows[0]]),
    )
    identities = load_ugi_training_identities(path, adapter, sha256_file(path))
    assert identities.provenance["non_train_rows_masked_before_structure_access"] == 1


@pytest.mark.parametrize("invalid_smiles", ["invalid molecular string", "C.C", ""])
@pytest.mark.parametrize("field_kind", ["product", "component"])
def test_malformed_training_structure_fails_closed(
    tmp_path, adapter, example_rows, invalid_smiles, field_kind
) -> None:
    field = "canonical_product_smiles" if field_kind == "product" else f"{adapter.roles[0]}_smiles"
    row = dict(example_rows[0], **{field: invalid_smiles})
    path = _write_assignments(tmp_path, [row])
    with pytest.raises(UgiTrainOnlyAssessmentError, match=field):
        load_ugi_training_identities(path, adapter, sha256_file(path))


def test_training_family_fold_mismatch_precedes_structure_parsing(
    tmp_path, adapter, example_rows
) -> None:
    field = f"{adapter.roles[0]}_family_fold"
    row = dict(example_rows[0], canonical_product_smiles="must not be parsed", **{field: "heldout"})
    path = _write_assignments(tmp_path, [row])
    with pytest.raises(UgiTrainOnlyAssessmentError, match=f"inconsistent {field}"):
        load_ugi_training_identities(path, adapter, sha256_file(path))


def test_missing_optional_family_fold_columns_remains_supported(tmp_path, adapter, example_rows):
    row = {key: value for key, value in example_rows[0].items() if not key.endswith("_family_fold")}
    path = _write_assignments(tmp_path, [row])
    identities = load_ugi_training_identities(path, adapter, sha256_file(path))
    assert identities.provenance["training_family_fold_checks_by_role"] == {
        role: 0 for role in adapter.roles
    }


def test_pin_mismatch_precedes_assignment_parsing(tmp_path, adapter, example_rows, monkeypatch):
    path = _write_assignments(tmp_path, [example_rows[0]])

    def forbidden_parse(_):
        raise AssertionError("mismatching assignment pin was parsed")

    monkeypatch.setattr("forge.model.ugi_train_only_assessment.iter_csv", forbidden_parse)
    with pytest.raises(PinError, match="changed"):
        load_ugi_training_identities(path, adapter, "0" * 64)


def test_nontraining_only_ledger_fails_empty_support(tmp_path, adapter, example_rows):
    path = _write_assignments(tmp_path, [dict(example_rows[0], primary_product_fold="heldout")])
    with pytest.raises(UgiTrainOnlyAssessmentError, match="support is empty"):
        load_ugi_training_identities(path, adapter, sha256_file(path))


def test_constitutional_identity_collapses_stereo_and_smiles_order(tmp_path, adapter, example_rows):
    row = example_rows[0]
    molecule = Chem.MolFromSmiles(row["canonical_product_smiles"])
    center, _ = Chem.FindMolChiralCenters(molecule, includeUnassigned=True)[0]
    atom = molecule.GetAtomWithIdx(center)
    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    first = dict(row, canonical_product_smiles=Chem.MolToSmiles(molecule, isomericSmiles=True))
    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    second = dict(row, canonical_product_smiles=Chem.MolToSmiles(molecule, isomericSmiles=True))
    assert first["canonical_product_smiles"] != second["canonical_product_smiles"]
    for role in adapter.roles:
        field = f"{role}_smiles"
        component = Chem.MolFromSmiles(row[field])
        second[field] = Chem.MolToSmiles(component, canonical=False, rootedAtAtom=1)
    path = _write_assignments(tmp_path, [first, second])
    identities = load_ugi_training_identities(path, adapter, sha256_file(path))
    assert identities.provenance["train_rows"] == 2
    assert len(identities.training_products) == 1
    assert all(len(values) == 1 for values in identities.training_components.values())
    _, report = assess_ugi_train_only_attempts(
        [_attempt(0, second["canonical_product_smiles"])],
        adapter=adapter,
        training_identities=identities,
    )
    assert report["metrics"]["whole_product_novel_to_train"] == 0
    assert report["metrics"]["decomposed_products_with_any_novel_component"] == 0


def test_failures_and_unresolved_products_keep_all_attempt_denominator(
    tmp_path, adapter, example_rows
):
    path = _write_assignments(tmp_path, [example_rows[0]])
    identities = load_ugi_training_identities(path, adapter, sha256_file(path))
    attempts = [
        _attempt(0, example_rows[0]["canonical_product_smiles"]),
        _attempt(1, example_rows[1]["canonical_product_smiles"]),
        _attempt(2, "C"),
        _attempt(3, "invalid molecular string"),
        _attempt(4, None, status="invalid"),
        _attempt(5, None, status="failed"),
    ]
    rows, report = assess_ugi_train_only_attempts(
        attempts, adapter=adapter, training_identities=identities
    )
    assert len(rows) == report["attempts"] == 6
    for attempt, row in zip(attempts, rows, strict=True):
        assert {key: row[key] for key in attempt.to_mapping()} == attempt.to_mapping()
    metrics = report["metrics"]
    assert (metrics["valid"], metrics["invalid_samples"], metrics["exact_l1_program"]) == (3, 3, 2)
    assert metrics["valid_fraction"] == 3 / 6
    assert metrics["exact_l1_yield_per_attempt"] == 2 / 6
    assert metrics["whole_product_novel_to_train_fraction"] == 2 / 3
    assert metrics["component_novelty_fraction"] == 1 / 2
    assert metrics["retro_decomposition_coverage_among_valid"] == 2 / 3
    assert metrics["retro_transform_precision"] == 1
    assert metrics["mean_pairwise_ecfp4_distance"] > 0
    incidence = report["all_attempt_metrics"]
    assert incidence["whole_product_novel_to_train"] == {
        "count": 2,
        "denominator": 6,
        "fraction": 2 / 6,
    }
    assert incidence["decomposed_products_with_any_novel_component"] == {
        "count": 1,
        "denominator": 6,
        "fraction": 1 / 6,
    }
    assert incidence["component_novelty_abstained"]["count"] == 4
    for role in adapter.roles:
        native = metrics["component_metrics_by_role"][role]
        assert native["slots"] == 2
        assert incidence["component_novel_to_train_by_role"][role] == {
            "count": native["novel"],
            "denominator": 6,
            "fraction": native["novel"] / 6,
        }
    assert report["coverage_and_precision_reported"] is True
    assert report["heldout_structures_interpreted"] is False
    assert report["candidate_selection"] is False


def test_ambiguous_exact_decomposition_abstains_without_dropping_attempt(
    tmp_path, adapter, example_rows, monkeypatch
):
    path = _write_assignments(tmp_path, [example_rows[0]])
    identities = load_ugi_training_identities(path, adapter, sha256_file(path))
    product = example_rows[0]["canonical_product_smiles"]
    traces = adapter.decompose(product)
    assert len(traces) == 1
    monkeypatch.setattr(Ugi3AssemblyAdapter, "decompose", lambda self, smiles: traces * 2)
    rows, report = assess_ugi_train_only_attempts(
        [_attempt(0, product), _attempt(1, None, status="failed")],
        adapter=adapter,
        training_identities=identities,
    )
    assert len(rows) == 2
    assert rows[0]["exact_l1_trace_count"] == 2
    assert report["metrics"]["exact_l1_program"] == 1
    assert report["metrics"]["ambiguous_exact_decompositions"] == 1
    assert report["metrics"]["component_novelty_fraction"] is None
    assert report["all_attempt_metrics"]["component_novelty_abstained"]["count"] == 2
    assert report["all_attempt_metrics"]["decomposed_products_with_any_novel_component"] == {
        "count": 0,
        "denominator": 2,
        "fraction": 0.0,
    }


def test_changed_adapter_reference_and_attempt_gaps_fail_closed(tmp_path, adapter, example_rows):
    path = _write_assignments(tmp_path, [example_rows[0]])
    identities = load_ugi_training_identities(path, adapter, sha256_file(path))
    attempts = [_attempt(0, example_rows[0]["canonical_product_smiles"])]
    with pytest.raises(UgiTrainOnlyAssessmentError, match="do not match"):
        assess_ugi_train_only_attempts(
            attempts,
            adapter=replace(adapter, registry_sha256="0" * 64),
            training_identities=identities,
        )
    with pytest.raises(CommonUgiBenchmarkError, match="every index"):
        assess_ugi_train_only_attempts(
            [replace(attempts[0], attempt_index=1)], adapter=adapter, training_identities=identities
        )
