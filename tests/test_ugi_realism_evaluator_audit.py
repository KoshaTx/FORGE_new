"""Behavioral checks for the supplemental, train-only evaluator audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.core.hashing import PinError, sha256_file
from forge.core.io import csv_gz_bytes, gzip_bytes, read_json_object, stable_json, write_json
from forge.model.common_lipid_realism import UGI_PRECURSOR_ROLES
from forge.model.ugi_realism_evaluator_audit import (
    FINGERPRINTS,
    RESULT_SCHEMA,
    AuditProduct,
    EvaluatorAuditError,
    canonical_graph,
    descriptor_collisions,
    drawing_variant,
    fingerprint_distance,
    load_train_reference,
    paired_summary,
    run_ugi_realism_evaluator_audit,
)

REPO = Path(__file__).resolve().parents[1]


def _pin(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": str(sha256_file(path))}


@pytest.fixture
def audit_fixture(tmp_path: Path) -> tuple[Path, Path]:
    config = read_json_object(REPO / "configs/multireaction/ugi_realism_evaluator_audit_v1.json")
    config.update(
        expected_attempts=3,
        invariance_pairs=1,
        positive_control_pairs=1,
        bootstrap_replicates=20,
        sources={},
    )
    reference = []
    for index, amine in enumerate(("NCCN(C)C", "NCCN(CC)CC", "NCCCN(C)C")):
        row = {
            "product_id": f"measured-{index}",
            "canonical_product_smiles": (
                "CCCCNC(=O)C(CCCOC(=O)CCC)" + ("NCCN(C)C", "NCCN(CC)CC", "NCCCN(C)C")[index]
            ),
            "primary_product_fold": "train",
            "source_stratum": "fixture_measured",
            "is_source_adjudicated_measured_product": "true",
            "component_novelty_class": "familiar_components",
        }
        for role, smiles in zip(
            UGI_PRECURSOR_ROLES, (amine, "CCCC(=O)OCCCC=O", "[C-]#[N+]CCCC"), strict=True
        ):
            row[f"{role}_smiles"] = smiles
            row[f"{role}_family_id"] = role
            row[f"{role}_family_fold"] = "train"
        reference.append(row)
    excluded = {key: "INVALID_HELDOUT_STRUCTURE_MUST_NOT_BE_PARSED" for key in reference[0]}
    excluded["primary_product_fold"] = "heldout"
    assignment_path = tmp_path / "assignments.csv.gz"
    assignment_path.write_bytes(csv_gz_bytes([*reference, excluded], list(reference[0])))
    pins = {"assignments": _pin(assignment_path, tmp_path)}
    methods = {}
    for arm, method in (("baseline", "amine_semantic"), ("treatment", "all_role_semantic")):
        records = [
            {
                "attempt_index": index,
                "canonical_smiles": row["canonical_product_smiles"],
                "exact_l1_program": True,
                "valid": True,
                "method_id": arm,
                "held_component_exact_l1": False,
                "exact_l1_traces": [
                    {
                        "components_by_role": {
                            role: row[f"{role}_smiles"] for role in UGI_PRECURSOR_ROLES
                        }
                    }
                ],
            }
            for index, row in enumerate(reference)
        ]
        path = tmp_path / f"{arm}.jsonl.gz"
        header = {"rows": 3, "schema_version": "forge.common_ugi_assessed_attempts.v1"}
        path.write_bytes(
            gzip_bytes(
                ("\n".join(stable_json(item) for item in [header, *records]) + "\n").encode()
            )
        )
        pins[arm] = _pin(path, tmp_path)
        common_path = tmp_path / f"{arm}_common_result.json"
        write_json(common_path, {"assessed_attempts": pins[arm], "method_id": arm})
        pins[f"{arm}_common_result"] = _pin(common_path, tmp_path)
        index_path = tmp_path / f"{arm}_assessment_index.json"
        write_json(index_path, {"assessment": {"common_assessment": pins[f"{arm}_common_result"]}})
        pins[f"{arm}_assessment_index"] = _pin(index_path, tmp_path)
        methods[method] = {"assessment": pins[f"{arm}_assessment_index"]}
    previous = tmp_path / "previous_review.json"
    write_json(
        previous,
        {
            "reviewer_id": "fixture",
            "reviews": [{"pair_id": "P01", "reviewer_note": "compact saturated silhouette"}],
        },
    )
    pins["previous_review"] = _pin(previous, tmp_path)
    comparison = tmp_path / "comparison.json"
    write_json(
        comparison, {"program_pairing": {"identical_coarse_programs": True}, "methods": methods}
    )
    pins["comparison"] = _pin(comparison, tmp_path)
    config["inputs"] = pins
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    return tmp_path, config_path


def test_constitution_isomorphism_and_positional_sensitivity() -> None:
    assert canonical_graph("NCCCO") == canonical_graph("OCCCN")
    for metric in FINGERPRINTS:
        assert fingerprint_distance("NCCCO", "OCCCN", metric) == 0.0
        assert fingerprint_distance("NCCCO", "NCCOC", metric) > 0.0


def test_descriptor_collision_is_real_graph_distinction() -> None:
    rows = []
    for index, aldehyde in enumerate(("C=CCCCCC(=O)OCCCCCC=O", "CC=CCCCC(=O)OCCCCCC=O")):
        roles = dict(
            zip(UGI_PRECURSOR_ROLES, ("NCCN(CC)CC", aldehyde, "[C-]#[N+]CCCC"), strict=True)
        )
        rows.append(
            AuditProduct(
                str(index),
                canonical_graph(aldehyde),
                tuple(roles.items()),
                "fixture",
                "family",
                "saved_fixture",
            )
        )
    result = descriptor_collisions(rows)
    assert result["collision_classes"] == 1
    assert result["unique_products_in_collision_classes"] == 2
    assert (
        result["groups"][0]["deterministic_first_pair_fingerprint_distances"]["atom_pair_count"] > 0
    )


def test_drawing_transform_preserves_graph_and_changes_svg() -> None:
    first, identity = drawing_variant("NCCCO", angle_degrees=0, reverse_atoms=False)
    second, rotated = drawing_variant("NCCCO", angle_degrees=117, reverse_atoms=True)
    assert identity == rotated
    assert first != second
    assert first == drawing_variant("NCCCO", angle_degrees=0, reverse_atoms=False)[0]


def test_run_determinism_provenance_and_pending_review(audit_fixture: tuple[Path, Path]) -> None:
    root, config = audit_fixture
    one = run_ugi_realism_evaluator_audit(root, config, root / "one")
    two = run_ugi_realism_evaluator_audit(root, config, root / "two")
    assert {key: value for key, value in one.items() if key != "created_at_utc"} == {
        key: value for key, value in two.items() if key != "created_at_utc"
    }
    assert one["created_at_utc"].endswith("+00:00")
    assert one["runtime"]["numpy_version"]
    assert one["schema_version"] == RESULT_SCHEMA
    assert one["numerical_complete"] is True
    assert one["reviewer_status"] == "pending"
    assert one["visual_calibration"]["new_review_outcomes"] is None
    assert one["train_reference"]["nontrain_rows_skipped_before_structure_access"] == 1
    assert one["config"]["sha256"] == sha256_file(config)
    assert all(value == 0 for value in one["calls"].values())
    assert (root / "one/review_packet.html").read_bytes() == (
        root / "two/review_packet.html"
    ).read_bytes()
    sheet = read_json_object(root / "one/review_sheet.json")
    assert sheet["reviewer_id"] is None
    assert all(row["overall_plausibility_preference"] is None for row in sheet["reviews"])
    key = read_json_object(root / "one/blinding_key.json")
    assert sum(row["kind"] == "identical_graph_redrawing" for row in key["pairs"]) == 1
    for pair in key["pairs"]:
        for slot in pair["slots"].values():
            assert slot["graph_identity_preserved"]
            assert slot["descriptor_identity_preserved"]
            assert slot["fingerprint_identity_preserved"]


def test_tampering_fails_before_output(audit_fixture: tuple[Path, Path]) -> None:
    root, config = audit_fixture
    (root / "previous_review.json").write_text("{}")
    with pytest.raises(PinError, match="changed"):
        run_ugi_realism_evaluator_audit(root, config, root / "bad")
    assert not (root / "bad").exists()


def test_provenance_chain_rejects_replaced_comparison(audit_fixture: tuple[Path, Path]) -> None:
    root, path = audit_fixture
    config = read_json_object(path)
    comparison_path = root / config["inputs"]["comparison"]["path"]
    comparison = read_json_object(comparison_path)
    comparison["methods"]["amine_semantic"]["assessment"]["sha256"] = "0" * 64
    write_json(comparison_path, comparison)
    config["inputs"]["comparison"] = _pin(comparison_path, root)
    write_json(path, config)
    with pytest.raises(EvaluatorAuditError, match="provenance chain"):
        run_ugi_realism_evaluator_audit(root, path, root / "bad")
    assert not (root / "bad").exists()


def test_train_family_mismatch_fails(audit_fixture: tuple[Path, Path]) -> None:
    root, _ = audit_fixture
    import csv
    import gzip

    with gzip.open(root / "assignments.csv.gz", "rt") as stream:
        rows = list(csv.DictReader(stream))
    rows[0][f"{UGI_PRECURSOR_ROLES[0]}_family_fold"] = "heldout"
    (root / "assignments.csv.gz").write_bytes(csv_gz_bytes(rows, list(rows[0])))
    with pytest.raises(EvaluatorAuditError, match="nontrain role family"):
        load_train_reference(root / "assignments.csv.gz")


def test_uncertainty_abstention_and_determinism() -> None:
    assert paired_summary([], seed=4, replicates=20)["status"] == "abstain_no_eligible_pairs"
    assert paired_summary([0.1, 0.2], seed=4, replicates=20) == paired_summary(
        [0.1, 0.2], seed=4, replicates=20
    )
