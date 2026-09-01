from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from experiments.phase1.multireaction.lipid_realism_aggregation import (
    aggregate_lipid_realism,
)
from experiments.phase1.multireaction.lipid_realism_assessment import (
    run_lipid_realism_assessment,
)
from forge.core.hashing import sha256_file
from forge.core.io import write_csv, write_json
from forge.model.common_lipid_realism import (
    RealismPolicy,
    assess_lipid_realism,
    build_ugi_development_realism_reference,
    build_ugi_realism_reference,
)
from forge.model.common_ugi_benchmark import CommonUgiAttempt, write_attempt_ledger
from forge.model.ugi_development_realism import (
    ROLE_DESCRIPTOR_NAMES,
    ugi_role_descriptor_vector,
)


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _write_reference(repo: Path) -> tuple[Path, Path]:
    train = [
        "CCCCCCCCNCCO",
        "CCCCCCCCCCNCCO",
        "CCCCCCN(CCO)CCCCCC",
        "CCCCCCCCN(CCO)CCCC",
        "CCCCCCOC(=O)CNCCCCC",
        "CCCCCCCCNC(=O)CO",
    ]
    heldout = [
        "CCCCCCCCCCNCCCO",
        "CCCCCCCN(CCO)CCCCCCC",
        "CCCCCCCCOC(=O)CNCCCC",
        "CCCCCCCCNCCCO",
        "CCCCCCN(CCCO)CCCCCC",
        "CCCCCCCCNC(=O)CCO",
    ]
    structures: list[dict[str, str]] = []
    assignments: list[dict[str, str]] = []
    for fold, molecules in (("R0_train", train), ("R0_heldout", heldout)):
        for index, smiles in enumerate(molecules):
            structure_id = f"{fold}:{index}"
            structures.append(
                {
                    "r0_structure_id": structure_id,
                    "canonical_constitutional_smiles": _canonical(smiles),
                    "r0_pretraining_eligible": "True",
                }
            )
            assignments.append(
                {
                    "r0_structure_id": structure_id,
                    "source_study_group_id": f"{fold}:group:{index % 3}",
                    "source_study_fold": fold,
                }
            )
    r0_path = repo / "r0.csv.gz"
    split_path = repo / "splits.csv"
    write_csv(r0_path, structures, list(structures[0]))
    write_csv(split_path, assignments, list(assignments[0]))
    return r0_path, split_path


def _write_ugi_reference(repo: Path) -> Path:
    train = [
        "CCCCCCCCNCCO",
        "CCCCCCCCCCNCCO",
        "CCCCCCN(CCO)CCCCCC",
        "CCCCCCCCN(CCO)CCCC",
        "CCCCCCOC(=O)CNCCCCC",
        "CCCCCCCCNC(=O)CO",
    ]
    calibration = [
        "CCCCCCCCCCNCCCO",
        "CCCCCCCN(CCO)CCCCCCC",
        "CCCCCCCCOC(=O)CNCCCC",
        "CCCCCCCCNCCCO",
        "CCCCCCN(CCCO)CCCCCC",
        "CCCCCCCCNC(=O)CCO",
    ]
    rows: list[dict[str, str]] = []
    for fold, molecules in (("train", train), ("calibration", calibration)):
        for index, smiles in enumerate(molecules):
            rows.append(
                {
                    "product_id": f"{fold}:{index}",
                    "canonical_product_smiles": _canonical(smiles),
                    "primary_product_fold": fold,
                    "is_source_adjudicated_measured_product": "true",
                    "amine_head_family_id": f"amine:{index % 3}",
                    "oxoester_aldehyde_body_tail_family_id": f"aldehyde:{index % 2}",
                    "isocyanide_tail_family_id": f"isocyanide:{index % 4}",
                }
            )
    rows.append(
        {
            "product_id": "virtual:0",
            "canonical_product_smiles": _canonical("CCCCCCCCCCCCNCCO"),
            "primary_product_fold": "calibration",
            "is_source_adjudicated_measured_product": "false",
            "amine_head_family_id": "amine:virtual",
            "oxoester_aldehyde_body_tail_family_id": "aldehyde:virtual",
            "isocyanide_tail_family_id": "isocyanide:virtual",
        }
    )
    path = repo / "ugi_assignments.csv.gz"
    write_csv(path, rows, list(rows[0]))
    return path


def _write_group_balanced_ugi_development_reference(repo: Path) -> Path:
    rows: list[dict[str, str]] = []
    for group in range(5):
        for member in range(4):
            chain = 4 + group * 4 + member
            rows.append(
                {
                    "product_id": f"train:{group}:{member}",
                    "canonical_product_smiles": _canonical("C" * chain + "NCCO"),
                    "primary_product_fold": "train",
                    "is_source_adjudicated_measured_product": "true",
                    "amine_head_family_id": f"amine:{group}",
                    "oxoester_aldehyde_body_tail_family_id": f"aldehyde:{group}",
                    "isocyanide_tail_family_id": f"isocyanide:{group}",
                    "amine_head_smiles": _canonical("NCCN(C)C"),
                    "oxoester_aldehyde_body_tail_smiles": _canonical(
                        "C" * (5 + group + member) + "C(=O)OCC=O"
                    ),
                    "isocyanide_tail_smiles": _canonical("[C-]#[N+]" + "C" * (5 + group + member)),
                }
            )
    for fold in ("calibration", "heldout"):
        rows.append(
            {
                "product_id": f"{fold}:unread",
                "canonical_product_smiles": "not-a-molecule",
                "primary_product_fold": fold,
                "is_source_adjudicated_measured_product": "true",
                "amine_head_family_id": "unread",
                "oxoester_aldehyde_body_tail_family_id": "unread",
                "isocyanide_tail_family_id": "unread",
                "amine_head_smiles": "not-a-component",
                "oxoester_aldehyde_body_tail_smiles": "not-a-component",
                "isocyanide_tail_smiles": "not-a-component",
            }
        )
    path = repo / "ugi_development_assignments.csv.gz"
    write_csv(path, rows, list(rows[0]))
    return path


def _policy() -> dict[str, object]:
    return {
        "allowed_elements": ["C", "N", "O", "P", "S"],
        "maximum_heavy_atoms": 194,
        "train_reference_limit": 6,
        "heldout_reference_limit": 6,
        "selection_seed": 11,
        "manifold_neighbors": 2,
        "fingerprint_radius": 2,
        "fingerprint_bits": 256,
        "distance_chunk_size": 2,
        "internal_diversity_limit": 6,
        "c2st_folds": 2,
        "c2st_minimum_rows_per_class": 2,
        "c2st_maximum_rows_per_class": 6,
        "c2st_max_iterations": 10,
        "c2st_seed": 13,
    }


def _attempts() -> list[CommonUgiAttempt]:
    products = [
        _canonical("CCCCCCCCCCNCCCO"),
        None,
        _canonical("C"),
        _canonical("CCCCCCCN(CCO)CCCCCCC"),
    ]
    statuses = ["generated", "failed", "generated", "generated"]
    return [
        CommonUgiAttempt(
            method_id="test_method",
            seed=17,
            attempt_index=index,
            status=status,  # type: ignore[arg-type]
            product_smiles=product,
            method_visible_component_ids=(),
            generator_calls=1,
            reaction_calls=0,
            route_calls=0,
            oracle_calls=0,
            wall_seconds=0.01,
        )
        for index, (status, product) in enumerate(zip(statuses, products, strict=True))
    ]


def test_realism_assessment_is_method_blind_deterministic_and_preserves_denominator(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    r0_path, split_path = _write_reference(repo)
    attempts_path = repo / "attempts.jsonl.gz"
    write_attempt_ledger(attempts_path, _attempts())
    config_path = repo / "config.json"
    write_json(
        config_path,
        {
            "schema_version": "forge.common_lipid_realism_config.v1",
            "scientific_question": "test",
            "inputs": {
                "r0_constitutional": {
                    "path": r0_path.name,
                    "sha256": str(sha256_file(r0_path)),
                },
                "r0_fold_assignments": {
                    "path": split_path.name,
                    "sha256": str(sha256_file(split_path)),
                },
            },
            "policy": _policy(),
            "metrics": {
                "attempt_denominator_includes_invalid_failed_duplicate_and_out_of_support": True,
                "fingerprint_and_descriptor_manifolds_reported_separately": True,
                "no_result_selected_thresholds": True,
                "qed_excluded": True,
            },
            "nonclaims": ["test nonclaim"],
        },
    )
    first = run_lipid_realism_assessment(
        config_path,
        repo,
        attempts_path,
        repo / "first",
        method_id="test_method",
        seed=17,
        expected_attempts=4,
    )
    second = run_lipid_realism_assessment(
        config_path,
        repo,
        attempts_path,
        repo / "second",
        method_id="test_method",
        seed=17,
        expected_attempts=4,
    )

    assessment = first["assessment"]
    assert first["status"] == "pass"
    assert first["gates"] == second["gates"]
    assert assessment == second["assessment"]
    assert first["assessed_attempts"]["sha256"] == second["assessed_attempts"]["sha256"]
    assert assessment["attempts"] == 4
    assert assessment["molecular_output"]["connected"] == 3
    assert assessment["molecular_output"]["connected_fraction_per_attempt"] == 0.75
    assert assessment["attempt_denominator_includes_invalid_failed_and_out_of_support"] is True
    assert assessment["coverage_and_precision_reported_separately"] is True
    assert assessment["qed_reported"] is False
    assert assessment["route_or_oracle_calls"] == 0
    assert set(assessment["empirical_lipid_manifold"]) == {
        "reference",
        "fingerprint",
        "descriptor",
        "nearest_reference_tanimoto",
        "nearest_reference_descriptor_distance",
        "normalized_descriptor_wasserstein",
        "classifier_two_sample",
    }


def test_realism_policy_rejects_a_reference_smaller_than_the_manifold_neighborhood() -> None:
    value = _policy()
    value["train_reference_limit"] = 2
    try:
        RealismPolicy.from_mapping(value)
    except ValueError as error:
        assert "exceed manifold_neighbors" in str(error)
    else:
        raise AssertionError("invalid reference policy was accepted")


def test_failed_attempts_never_enter_the_structural_manifold(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    r0_path, split_path = _write_reference(repo)
    from forge.model.common_lipid_realism import build_realism_reference

    policy = RealismPolicy.from_mapping(_policy())
    reference = build_realism_reference(r0_path, split_path, policy)
    assert build_realism_reference(r0_path, split_path, policy) is reference
    rows, result = assess_lipid_realism(_attempts(), reference, policy)
    assert rows[1]["native_status"] == "failed"
    assert rows[1]["connected"] is False
    assert rows[1]["fingerprint_manifold_member"] is False
    assert rows[1]["descriptor_manifold_member"] is False
    assert (
        result["empirical_lipid_manifold"]["fingerprint"]["precision_per_requested_attempt"] <= 0.75
    )


def test_ugi_matched_reference_uses_only_measured_rows_and_preserves_the_fold(
    tmp_path: Path,
) -> None:
    assignments = _write_ugi_reference(tmp_path)
    policy = RealismPolicy.from_mapping(_policy())
    reference = build_ugi_realism_reference(
        assignments,
        policy,
        evaluation_fold="calibration",
    )

    assert len(reference.training) == 6
    assert len(reference.heldout) == 6
    assert all(row.structure_id.startswith("train:") for row in reference.training)
    assert all(row.structure_id.startswith("calibration:") for row in reference.heldout)
    assert reference.audit["virtual_products_used_as_realism_reference"] is False
    assert reference.audit["source_counts"]["excluded_not_source_adjudicated_measured"] == 1
    assert reference.audit["evaluation_fold"] == "calibration"
    assert reference.audit["scaling_population"] == (
        "source-adjudicated measured Ugi train fold only"
    )


def test_ugi_development_reference_is_group_balanced_and_never_parses_nontrain_products(
    tmp_path: Path,
) -> None:
    assignments = _write_group_balanced_ugi_development_reference(tmp_path)
    policy_value = _policy()
    policy_value["train_reference_limit"] = 10
    policy_value["heldout_reference_limit"] = 10
    policy_value["c2st_maximum_rows_per_class"] = 10
    reference = build_ugi_development_realism_reference(
        assignments,
        RealismPolicy.from_mapping(policy_value),
        rows_per_group_per_partition=1,
        split_seed=29,
    )

    assert len(reference.scaling) == 5
    assert len(reference.evaluation) == 5
    assert len({row.group_id for row in reference.scaling}) == 5
    assert len({row.group_id for row in reference.evaluation}) == 5
    assert reference.realism.audit["calibration_or_heldout_product_structures_accessed"] is False
    assert (
        reference.realism.audit["source_counts"][
            "ignored_measured_non_train_before_structure_access"
        ]
        == 2
    )


def test_ugi_role_descriptor_exposes_tail_ring_size_and_oxygen_rings() -> None:
    vector = ugi_role_descriptor_vector(
        {
            "amine_head": "NCCN(C)C",
            "oxoester_aldehyde_body_tail": "O=CC1CCCCCO1",
            "isocyanide_tail": "[C-]#[N+]CCCCCCCC",
        }
    )
    by_name = dict(zip(ROLE_DESCRIPTOR_NAMES, vector, strict=True))

    assert by_name["aldehyde_ring_count"] == 1.0
    assert by_name["aldehyde_largest_ring_size"] == 7.0
    assert by_name["aldehyde_oxygen_ring_count"] == 1.0


def _minimal_seed_result(method: str, seed: int, connected: float) -> dict[str, object]:
    molecular = {
        "connected_fraction_per_attempt": connected,
        "unique_fraction_among_connected": connected,
        "effective_molecule_count": connected * 100,
        "within_declared_support_fraction_per_attempt": connected,
        "mean_pairwise_ecfp4_distance_among_unique": connected,
    }
    manifold = {
        "fingerprint": {
            "precision_per_requested_attempt": connected,
            "coverage": connected,
        },
        "descriptor": {
            "precision_per_requested_attempt": connected,
            "coverage": connected,
        },
        "normalized_descriptor_wasserstein": {"mean_across_descriptors": 1.0 - connected},
        "classifier_two_sample": {"auc_mean": 1.0 - connected / 2},
    }
    return {
        "schema_version": "forge.common_lipid_realism_complete_assessment.v1",
        "status": "pass",
        "method_id": method,
        "seed": seed,
        "config": {"sha256": "a" * 64},
        "inputs": {
            "r0_constitutional": {"sha256": "b" * 64},
            "r0_fold_assignments": {"sha256": "c" * 64},
        },
        "gates": {"all": True},
        "assessment": {
            "method_id": method,
            "seed": seed,
            "attempts": 100,
            "molecular_output": molecular,
            "empirical_lipid_manifold": manifold,
        },
    }


def test_realism_aggregate_uses_seed_mean_and_sample_standard_deviation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    result_paths = []
    for seed, connected in ((1, 0.2), (2, 0.4), (3, 0.6)):
        path = repo / f"seed{seed}.json"
        write_json(path, _minimal_seed_result("method", seed, connected))
        result_paths.append(path)

    result = aggregate_lipid_realism(
        result_paths,
        repo,
        repo / "aggregate",
        expected_seeds=(1, 2, 3),
    )
    summary = result["methods"]["method"]["metrics"]["connected_fraction_per_attempt"]
    assert abs(summary["mean"] - 0.4) < 1e-12
    assert abs(summary["sample_standard_deviation"] - 0.2) < 1e-12
    assert result["gates"]["training_seed_is_independent_unit"] is True
    assert result["gates"]["molecule_rows_not_treated_as_replicates"] is True
