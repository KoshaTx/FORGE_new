from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/phase1_select_ugi_architecture.py"
REPO = SCRIPT.parents[1]
SPEC = importlib.util.spec_from_file_location("phase1_select_ugi_architecture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _policy() -> dict:
    return {
        "matched_uncertainty": {
            "seed": 7,
            "bootstrap_replicates": 1000,
            "confidence_interval": 0.95,
            "operationally_meaningful_absolute_difference": 0.02,
        }
    }


def test_empirical_distribution_distances_are_exact_on_separated_samples() -> None:
    left = np.asarray([0.0, 1.0])
    right = np.asarray([1.0, 2.0])
    assert MODULE._wasserstein_1(left, right) == 1.0
    assert MODULE._ks_distance(np.asarray([0.0, 0.0]), np.asarray([1.0, 1.0])) == 1.0


def test_paired_bootstrap_preserves_constant_difference() -> None:
    comparison = MODULE._bootstrap_difference(
        np.ones(32, dtype=np.int8),
        np.zeros(32, dtype=np.int8),
        seed=11,
        replicates=500,
        confidence=0.95,
    )
    assert comparison == {"point_difference": 1.0, "lower": 1.0, "upper": 1.0}


def test_within_architecture_uses_primary_endpoint_before_realism() -> None:
    candidates = [
        {
            "candidate_id": "arm:step_250",
            "step": 250,
            "gate": {"status": "pass"},
            "usable_open_ended_yield": 1.0,
            "mean_normalized_wasserstein_1": 0.20,
            "exact_frozen_product_fraction_among_valid": 0.05,
        },
        {
            "candidate_id": "arm:step_500",
            "step": 500,
            "gate": {"status": "pass"},
            "usable_open_ended_yield": 0.0,
            "mean_normalized_wasserstein_1": 0.01,
            "exact_frozen_product_fraction_among_valid": 0.0,
        },
    ]
    vectors = {
        "arm:step_250": np.ones(32, dtype=np.int8),
        "arm:step_500": np.zeros(32, dtype=np.int8),
    }
    selected = MODULE._within_architecture_selection(candidates, vectors, _policy())
    assert selected["selected"] == "arm:step_250"
    assert selected["noninferior_candidates"] == ["arm:step_250"]


def test_within_architecture_uses_realism_for_noninferior_yields() -> None:
    candidates = [
        {
            "candidate_id": "arm:step_250",
            "step": 250,
            "gate": {"status": "pass"},
            "usable_open_ended_yield": 0.50,
            "mean_normalized_wasserstein_1": 0.20,
            "exact_frozen_product_fraction_among_valid": 0.02,
        },
        {
            "candidate_id": "arm:step_500",
            "step": 500,
            "gate": {"status": "pass"},
            "usable_open_ended_yield": 0.50,
            "mean_normalized_wasserstein_1": 0.10,
            "exact_frozen_product_fraction_among_valid": 0.03,
        },
    ]
    vector = np.asarray([0, 1] * 16, dtype=np.int8)
    vectors = {candidate["candidate_id"]: vector for candidate in candidates}
    selected = MODULE._within_architecture_selection(candidates, vectors, _policy())
    assert selected["selected"] == "arm:step_500"


def test_noncanonical_smiles_use_canonical_membership_key() -> None:
    _, product = MODULE._canonicalized_molecule("C(C)O")
    _, component = MODULE._canonicalized_molecule("NCC")
    assert product == "CCO"
    assert component == "CCN"
    assert product in {"CCO"}
    assert component in {"CCN"}


def test_selection_reference_excludes_heldout_fold() -> None:
    assignments = [
        {"primary_product_fold": "train", "canonical_product_smiles": "C"},
        {"primary_product_fold": "calibration", "canonical_product_smiles": "CC"},
        {"primary_product_fold": "heldout", "canonical_product_smiles": "CCC"},
    ]
    selected, counts = MODULE._selection_reference_rows(
        assignments,
        {
            "selection_folds": ["train", "calibration"],
            "selection_rows": 2,
        },
    )
    assert counts == {"train": 1, "calibration": 1, "heldout": 1}
    assert [row["canonical_product_smiles"] for row in selected] == ["C", "CC"]


def test_metadata_bearing_reference_exposes_only_path_and_hash() -> None:
    specification = MODULE._path_hash_specification(
        {
            "path": "reference.csv.gz",
            "sha256": "a" * 64,
            "rows": 112386,
            "selection_folds": ["train", "calibration"],
        }
    )
    assert specification == {"path": "reference.csv.gz", "sha256": "a" * 64}


def test_nested_policy_inheritance_resolves_every_base(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps({"schema_version": "base", "scope": {"a": 1}}))
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    middle_path = tmp_path / "middle.json"
    middle_path.write_text(
        json.dumps(
            {
                "schema_version": "middle",
                "base_policy": {"path": str(base_path), "sha256": base_hash},
                "overrides": {"scope": {"b": 2}},
            }
        )
    )
    middle_hash = hashlib.sha256(middle_path.read_bytes()).hexdigest()
    top_path = tmp_path / "top.json"
    top_path.write_text(
        json.dumps(
            {
                "schema_version": "top",
                "base_policy": {"path": str(middle_path), "sha256": middle_hash},
                "overrides": {"scope": {"c": 3}},
            }
        )
    )
    revision, resolved = MODULE._load_selection_policy(top_path)
    assert revision["schema_version"] == "top"
    assert resolved["schema_version"] == "top"
    assert resolved["scope"] == {"a": 1, "b": 2, "c": 3}


def test_checkpoint_input_contract_rejects_atom_vocabulary_drift() -> None:
    expected = {
        key: {"path": f"{key}.dat", "sha256": "a" * 64} for key in MODULE.COMMON_TRAINING_INPUT_KEYS
    }
    checkpoint = {
        "inputs": {
            key: {"path": str(MODULE.REPO / value["path"]), "sha256": value["sha256"]}
            for key, value in expected.items()
        }
    }
    MODULE._require_checkpoint_inputs(
        checkpoint,
        expected,
        architecture="full_morphology_program_conditioning",
        step=250,
    )
    checkpoint["inputs"]["atom_vocabulary"]["sha256"] = "b" * 64
    try:
        MODULE._require_checkpoint_inputs(
            checkpoint,
            expected,
            architecture="full_morphology_program_conditioning",
            step=250,
        )
    except MODULE.SelectionPolicyError as error:
        assert "checkpoint embedded input path/hash mismatch" in str(error)
    else:
        raise AssertionError("atom-vocabulary drift was not rejected")


def test_sampling_contract_binds_production_decoder_and_branch_limits() -> None:
    policy = {
        "matched_sampling": {
            "sample_steps": 8,
            "maximum_adjacent_branch_graph_runs_by_role": {
                "amine_head": 2,
                "oxoester_aldehyde_body_tail": 1,
                "isocyanide_tail": 1,
            },
            "evaluate_exact_l1_terminal_admission": True,
            "terminal_decoder_mode": "bond_stochastic",
            "terminal_decoder_seed": 17,
            "terminal_temperature": 1.0,
        }
    }
    result = {
        "sampling": {
            "sample_steps": 8,
            "maximum_adjacent_branch_runs": [2, 1, 1],
            "evaluate_exact_l1_terminal_admission": True,
            "terminal_decoder": {
                "mode": "bond_stochastic",
                "seed": 17,
                "temperature": 1.0,
            },
        }
    }
    MODULE._require_sampling_contract(
        result,
        policy,
        architecture="full_morphology_program_conditioning",
        step=1000,
    )
    result["sampling"]["terminal_decoder"]["mode"] = "argmax"
    try:
        MODULE._require_sampling_contract(
            result,
            policy,
            architecture="full_morphology_program_conditioning",
            step=1000,
        )
    except MODULE.SelectionPolicyError as error:
        assert "terminal decoder changed" in str(error)
    else:
        raise AssertionError("terminal decoder drift was not rejected")


def test_program_draw_seed_is_independent_of_flow_sampling_seed() -> None:
    policy = {
        "matched_uncertainty": {"seed": 5},
        "matched_sampling": {"program_seed": 11, "seed": 17},
    }
    assert MODULE._expected_program_seed(policy) == 11

    legacy_policy = {
        "matched_uncertainty": {"seed": 5},
        "matched_sampling": {"seed": 17},
    }
    assert MODULE._expected_program_seed(legacy_policy) == 17


def test_branch_spacing_promotion_changes_scope_but_not_v3_gates() -> None:
    _, v3 = MODULE._load_selection_policy(
        REPO / "configs/model/phase1_ugi_architecture_checkpoint_selection_policy_v3.json"
    )
    _, promotion = MODULE._load_selection_policy(
        REPO / "configs/model/phase1_ugi_branch_spacing_checkpoint_promotion_policy_v1.json"
    )
    unchanged_blocks = (
        "reference_inverse_gate",
        "candidate_hard_gates",
        "per_role_component_collapse_safeguards",
        "descriptor_distribution_realism_floors",
        "primary_selection_statistic",
        "matched_uncertainty",
        "selection_algorithm",
    )
    for key in unchanged_blocks:
        assert promotion[key] == v3[key]

    assert promotion["scope"]["architectures"] == ["full_morphology_program_conditioning"]
    assert promotion["scope"]["common_serial_checkpoint_steps"] == [1000, 2000]
    assert promotion["matched_sampling"]["maximum_adjacent_branch_graph_runs_by_role"] == {
        "amine_head": 2,
        "oxoester_aldehyde_body_tail": 1,
        "isocyanide_tail": 1,
    }
