"""Behavior tests for train-only support attribution and unchanged reference controls."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forge.core.hashing import PinError, sha256_file
from forge.model.common_lipid_realism import RealismPolicy, UgiMeasuredReferenceMolecule
from forge.model.ugi_ester_chemotype import UgiEsterChemotypePolicy
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.model.ugi_realism_support_audit import (
    UgiRealismSupportAuditError,
    _equal_group_matrix,
    assess_support_reference_controls,
    program_exclusion_reasons,
    run_ugi_realism_support_audit,
    select_measured_training_rows,
    summarize_support,
)
from forge.potency.annotations import ROLE_NAMES

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def ester_policy(tmp_path: Path) -> UgiEsterChemotypePolicy:
    return UgiEsterChemotypePolicy(
        reaction_id="ugi_3cr_agile",
        amine_role=ROLE_NAMES[0],
        aldehyde_role=ROLE_NAMES[1],
        isocyanide_role=ROLE_NAMES[2],
        registry_path=tmp_path / "registry.json",
        registry_sha256="0" * 64,
        training_assignments_path=tmp_path / "assignments.csv.gz",
        training_assignments_sha256="0" * 64,
        minimum_role_exterior_atoms=tuple(zip(ROLE_NAMES, (3, 20, 12), strict=True)),
        maximum_role_exterior_atoms=tuple(zip(ROLE_NAMES, (8, 30, 16), strict=True)),
        minimum_ester_side_carbons=6,
        minimum_ester_long_side_carbons=10,
        minimum_amine_exterior_nitrogens=1,
        maximum_amine_exterior_nitrogens=2,
        morphology_quantile=0.25,
    )


def _program() -> UgiMorphologyProgram:
    return UgiMorphologyProgram((5, 25, 14), (1, 1, 0), (0, 0, 0))


def _assignment(product_id: str = "a") -> dict[str, str]:
    row = {
        "primary_product_fold": "train",
        "is_source_adjudicated_measured_product": "true",
        "product_id": product_id,
        "canonical_product_smiles": "CCCC",
    }
    for role, smiles in zip(
        ROLE_NAMES, ("CNCCN", "O=CCCCOC(=O)CCCC", "CCCC[N+]#[C-]"), strict=True
    ):
        row[f"{role}_smiles"] = smiles
        row[f"{role}_family_id"] = f"{role}_family"
    return row


def _ledger_row(product_id: str, *, reasons: tuple[str, ...] = ()) -> dict:
    selected, _ = select_measured_training_rows([_assignment(product_id)])
    return {
        **selected[product_id],
        "program_admitted": not reasons,
        "exclusion_reasons": list(reasons),
    }


def test_fold_masking_precedes_every_structure_access() -> None:
    class SealedRow(dict):
        def __getitem__(self, key: str):
            if key != "primary_product_fold":
                raise AssertionError(f"non-train field accessed: {key}")
            return super().__getitem__(key)

    selected, counts = select_measured_training_rows(
        [
            SealedRow(primary_product_fold="heldout"),
            SealedRow(primary_product_fold="calibration"),
            _assignment(),
        ]
    )
    assert list(selected) == ["a"]
    assert counts["non_train_rows_masked_before_structure_access"] == 2


def test_nonmeasured_mask_precedes_structure_access() -> None:
    selected, counts = select_measured_training_rows(
        [
            {"primary_product_fold": "train", "is_source_adjudicated_measured_product": "false"},
            _assignment(),
        ]
    )
    assert len(selected) == 1
    assert counts["non_measured_train_rows_masked_before_structure_access"] == 1


def test_duplicate_measurement_rejected() -> None:
    with pytest.raises(UgiRealismSupportAuditError, match="duplicate measured train product"):
        select_measured_training_rows([_assignment(), _assignment()])


def test_gate_attribution_reports_every_overlapping_reason(ester_policy) -> None:
    assert program_exclusion_reasons(_program(), ester_policy) == []
    bad = replace(_program(), node_counts=(2, 35, 14), junction_budgets=(1, 0, 0))
    assert program_exclusion_reasons(bad, ester_policy) == sorted(
        [
            f"{ROLE_NAMES[0]}.below_minimum_exterior_atoms",
            f"{ROLE_NAMES[1]}.above_maximum_exterior_atoms",
            "existing_decoder_non_count_topology_restriction",
        ]
    )


def test_attribution_partition_conserves_products_and_components() -> None:
    rows = [
        _ledger_row("a"),
        _ledger_row("b", reasons=("small_head", "large_tail")),
        _ledger_row("c", reasons=("large_tail",)),
    ]
    rows[-1]["components_by_role"][ROLE_NAMES[0]] = "NCCCCN"
    summary = summarize_support(rows)
    assert summary["admitted_products"] == 1
    assert summary["excluded_products"] == 2
    assert sum(summary["mutually_exclusive_reason_patterns"].values()) == 3
    assert summary["exclusion_reasons_overlapping"] == {"large_tail": 2, "small_head": 1}
    amines = summary["unique_components_by_role"][ROLE_NAMES[0]]
    assert (amines["observed"], amines["entirely_excluded"], amines["partially_excluded"]) == (
        2,
        1,
        1,
    )
    assert summarize_support(list(reversed(rows))) == summary


def test_contradictory_admission_rejected() -> None:
    row = _ledger_row("a")
    row["exclusion_reasons"] = ["excluded"]
    with pytest.raises(UgiRealismSupportAuditError, match="inconsistent admission"):
        summarize_support([row])


def test_equal_group_mass_is_exact_and_bounded() -> None:
    rows = [{"group_id": "one"}, {"group_id": "two"}, {"group_id": "two"}]
    matrix, audit = _equal_group_matrix(rows, np.asarray([[0.0], [10.0], [20.0]]), maximum_rows=4)
    assert matrix[:, 0].tolist() == [0.0, 0.0, 10.0, 20.0]
    assert float(matrix.mean()) == 7.5
    assert audit["independent_observations_added"] == 0
    with pytest.raises(UgiRealismSupportAuditError, match="above 3"):
        _equal_group_matrix(rows, np.zeros((3, 1)), maximum_rows=3)


def test_reference_controls_are_deterministic_and_preserve_full_reference() -> None:
    rows = [_ledger_row("a"), _ledger_row("b", reasons=("excluded",))]
    rows[1]["components_by_role"][ROLE_NAMES[0]] = "NCCCCN"
    reference_rows = tuple(
        UgiMeasuredReferenceMolecule(
            row["product_id"],
            row["canonical_product_smiles"],
            row["group_id"],
            tuple(row["components_by_role"].items()),
        )
        for row in rows
    )
    reference = SimpleNamespace(
        scaling=reference_rows,
        evaluation=reference_rows,
        realism=SimpleNamespace(audit={"selection_sha256": "unchanged"}),
    )
    config = json.loads(
        (REPO / "configs/multireaction/ugi_development_lipid_realism_v3.json").read_text()
    )
    policy = RealismPolicy.from_mapping(config["policy"])
    first = assess_support_reference_controls(rows, reference, policy, maximum_replicated_rows=10)
    second = assess_support_reference_controls(rows, reference, policy, maximum_replicated_rows=10)
    assert first == second
    assert first["counts"]["full_evaluation_rows"] == 2
    assert first["counts"]["admitted_evaluation_rows"] == 1
    assert first["full_reference"] == {"selection_sha256": "unchanged"}
    assert first["frozen_reference_or_gate_changed"] is False
    assert first["possible_generator_best_case_bound"] is False
    assert (
        first["full_reference_diagnostic"]["admitted_measured_product_uniform_vs_full_evaluation"][
            "energy_distance"
        ]
        > 0
    )
    assert (
        first["supplemental_admitted_reference_diagnostic"][
            "admitted_measured_product_uniform_vs_admitted_evaluation"
        ]["energy_distance"]
        == 0
    )


def test_input_tampering_fails_before_opening_cache_or_publishing(tmp_path, monkeypatch) -> None:
    config = json.loads(
        (REPO / "configs/multireaction/ugi_realism_support_audit_v1.json").read_text()
    )
    payload = tmp_path / "input.json"
    payload.write_text("{}")
    pin = {"path": payload.name, "sha256": str(sha256_file(payload))}
    config["inputs"] = dict.fromkeys(config["inputs"], pin)
    config["sources"] = dict.fromkeys(config["sources"], pin)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    payload.write_text('{"tampered":true}')

    def fail_cache(*args, **kwargs):
        raise AssertionError("cache opened before provenance validation")

    monkeypatch.setattr(
        "forge.model.ugi_realism_support_audit.SynthesisProgramProductionCache", fail_cache
    )
    with pytest.raises(PinError, match="changed"):
        run_ugi_realism_support_audit(tmp_path, path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_existing_output_is_never_overwritten(tmp_path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(UgiRealismSupportAuditError, match="already exists"):
        run_ugi_realism_support_audit(tmp_path, tmp_path / "missing_config.json", output)
