from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from rdkit import Chem

from forge.route.engine.single_step_benchmark_manifest import (
    OUTPUT_FILENAMES,
    SingleStepBenchmarkManifestError,
    _load_role_queries,
    _PinnedReader,
    _qualifies_role,
    _round_robin_select,
    build_manifest_payloads,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/single_step_proposal_lane_qualification_manifest_v1.json"
FINAL_CONFIG = REPO / "configs/route/single_step_proposal_lane_qualification_benchmark_v2.json"

EXPECTED_STRATA = {
    "known_exact_l2_routes": 18,
    "held_reaction_families": 18,
    "linear_aldehydes": 12,
    "branched_aldehydes": 12,
    "unsaturated_aldehydes": 12,
    "ester_containing_aldehydes": 12,
    "isocyanide_formamide_precursors": 12,
    "heterocyclic_amine_heads": 12,
    "adversarial_incompatibles": 12,
}


def _decode(payload: bytes, *, compressed: bool = True) -> dict:
    raw = gzip.decompress(payload) if compressed else payload
    return json.loads(raw)


@pytest.fixture(scope="module")
def built() -> dict[str, bytes]:
    return build_manifest_payloads(REPO, CONFIG)


def test_real_manifest_is_deterministic_disjoint_and_truth_separated(
    built: dict[str, bytes],
) -> None:
    repeated = build_manifest_payloads(REPO, CONFIG)
    assert built == repeated
    assert set(built) == set(OUTPUT_FILENAMES)

    lane = _decode(built["lane_targets"])
    truth = _decode(built["scoring_truth"])
    masks = _decode(built["visibility_masks"])
    result = _decode(built["result"], compressed=False)

    targets = lane["targets"]
    assert len(targets) == 120
    assert Counter(row["primary_stratum"] for row in targets) == Counter(EXPECTED_STRATA)
    assert len({row["target_id"] for row in targets}) == 120
    assert len({row["canonical_smiles"] for row in targets}) == 120
    assert lane["truth_fields_present"] is False
    assert lane["known_routes_exposed_to_lanes"] is False
    assert all("reactant" not in json.dumps(row).lower() for row in targets)

    assert len(truth["records"]) == 48
    assert (
        sum(
            row["truth_kind"] == "documented_exact_forward_unique_reactant_multiset"
            for row in truth["records"]
        )
        == 36
    )
    assert (
        sum(
            row["truth_kind"] == "valid_connected_wrong_handle_role_swap_control"
            for row in truth["records"]
        )
        == 12
    )
    assert len(masks["masks"]) == 36
    assert result["summary"]["primary_strata"] == EXPECTED_STRATA
    assert result["summary"]["constitutional_unique_targets"] == 120
    assert result["safety_receipt"]["sealed_holdout_accessed"] is False
    assert result["safety_receipt"]["forbidden_inputs_accessed"] == []
    assert result["safety_receipt"]["graph2edits_executed"] is False
    assert result["safety_receipt"]["proposal_lane_executed"] is False
    for label in ("lane_targets", "scoring_truth", "visibility_masks"):
        assert result["outputs"][label]["sha256"] == hashlib.sha256(built[label]).hexdigest()


def test_every_target_is_a_valid_connected_constitution(built: dict[str, bytes]) -> None:
    lane = _decode(built["lane_targets"])
    for row in lane["targets"]:
        molecule = Chem.MolFromSmiles(row["canonical_smiles"])
        assert molecule is not None
        assert len(Chem.GetMolFrags(molecule)) == 1
        assert (
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
            == row["canonical_smiles"]
        )


def test_adversarial_controls_are_valid_wrong_role_assignments(
    built: dict[str, bytes],
) -> None:
    lane = _decode(built["lane_targets"])
    truth = _decode(built["scoring_truth"])
    targets = {row["target_id"]: row for row in lane["targets"]}
    registry = json.loads((REPO / "data/vendor/qualified_reactions_v1.json").read_text())
    queries = _load_role_queries(registry)

    controls = [
        row
        for row in truth["records"]
        if row["truth_kind"] == "valid_connected_wrong_handle_role_swap_control"
    ]
    assert len(controls) == 12
    for control in controls:
        target = targets[control["target_id"]]
        molecule = Chem.MolFromSmiles(target["canonical_smiles"])
        assert molecule is not None
        assert _qualifies_role(molecule, control["source_qualified_role"], queries)
        assert not _qualifies_role(molecule, control["adversarial_assigned_role"], queries)
        assert target["declared_role"] == control["adversarial_assigned_role"]


def test_visibility_masks_hide_exact_truth_and_complete_held_families(
    built: dict[str, bytes],
) -> None:
    lane = _decode(built["lane_targets"])
    truth = _decode(built["scoring_truth"])
    masks = _decode(built["visibility_masks"])
    target_by_id = {row["target_id"]: row for row in lane["targets"]}
    truth_by_id = {
        row["target_id"]: row
        for row in truth["records"]
        if row["truth_kind"] == "documented_exact_forward_unique_reactant_multiset"
    }

    assert {row["target_id"] for row in masks["masks"]} == set(truth_by_id)
    assert masks["graph2edits_pretraining_family_disjointness_claimed"] is False
    held_families: set[str] = set()
    for mask in masks["masks"]:
        target = target_by_id[mask["target_id"]]
        truth_row = truth_by_id[mask["target_id"]]
        assert mask["exact_reactants_exposed_to_lane"] is False
        if target["primary_stratum"] == "held_reaction_families":
            assert mask["hidden_local_transformation_families"] == [truth_row["transformation"]]
            assert mask["hidden_local_qualified_reaction_ids"]
            assert mask["hidden_local_record_key_sha256s"]
            held_families.add(truth_row["transformation"])
        else:
            assert target["primary_stratum"] == "known_exact_l2_routes"
            assert mask["hidden_local_transformation_families"] == []
            assert mask["hidden_local_qualified_reaction_ids"] == []
            assert mask["hidden_local_record_key_sha256s"] == [truth_row["local_record_key_sha256"]]
    assert held_families == {
        "alcohol_to_aldehyde_oxidation",
        "amine_formylation",
        "esterification",
        "formamide_dehydration_to_isocyanide",
    }


def test_forbidden_reader_rejects_before_access(tmp_path: Path) -> None:
    forbidden = tmp_path / "sealed" / "private.json"
    forbidden.parent.mkdir()
    forbidden.write_text("{}\n")
    reader = _PinnedReader(
        tmp_path,
        {
            "private": {
                "path": "sealed/private.json",
                "sha256": hashlib.sha256(forbidden.read_bytes()).hexdigest(),
            }
        },
        ("sealed",),
    )
    with pytest.raises(SingleStepBenchmarkManifestError, match="forbidden source"):
        reader.json("private")
    assert reader.accessed == {}


def test_round_robin_selection_fails_instead_of_substituting() -> None:
    records = [
        {"transformation": "family_a", "canonical_smiles": "CC"},
        {"transformation": "family_b", "canonical_smiles": "CCC"},
    ]
    with pytest.raises(SingleStepBenchmarkManifestError, match="only 2 honest route targets"):
        _round_robin_select(records, count=3, seed=7, namespace="insufficient")


def test_final_binding_config_authenticates_frozen_outputs_and_remains_inactive() -> None:
    config = json.loads(FINAL_CONFIG.read_text())
    assert config["status"] == "frozen_manifest_bound_benchmark_not_executed"
    for record in (
        config["frozen_policy"],
        config["manifest_builder_contract"],
        *config["development_target_manifest"].values(),
    ):
        path = REPO / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert config["manifest_summary"]["target_count"] == 120
    assert config["manifest_summary"]["constitutional_unique_targets"] == 120
    assert config["manifest_summary"]["sealed_or_private_inputs_accessed"] is False
    assert config["prerequisite_gates"]["development_target_manifest_frozen"] is True
    assert config["prerequisite_gates"]["all_gates_passed"] is False
    assert config["activation"] == {
        "benchmark_execution_authorized": False,
        "graph2edits_backend_active": False,
        "hybrid_production_lane_active": False,
        "strict_lane_remains_production_default": True,
        "candidate_selection_authorized": False,
    }
