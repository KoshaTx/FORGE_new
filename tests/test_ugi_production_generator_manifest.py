from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "results/phase1/ugi_product_l1_production_generator_v1.json"
MANIFEST_V2 = REPO / "results/phase1/ugi_product_l1_production_generator_v2.json"
MANIFEST_V3 = REPO / "results/phase1/ugi_product_l1_production_generator_v3.json"


def _artifact_specifications(value: object) -> list[dict[str, str]]:
    specifications: list[dict[str, str]] = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            specifications.append({"path": value["path"], "sha256": value["sha256"]})
        for child in value.values():
            specifications.extend(_artifact_specifications(child))
    elif isinstance(value, list):
        for child in value:
            specifications.extend(_artifact_specifications(child))
    return specifications


def test_production_generator_manifest_authenticates_all_pinned_artifacts() -> None:
    manifest = json.loads(MANIFEST.read_text())
    specifications = _artifact_specifications(manifest)
    assert len(specifications) == 16
    for specification in specifications:
        path = REPO / specification["path"]
        assert path.is_file(), specification["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == specification["sha256"]


def test_production_generator_manifest_freezes_product_plus_l1_scope() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["status"] == "frozen"
    assert manifest["identity"]["architecture"] == "full_morphology_program_conditioning"
    assert manifest["identity"]["checkpoint_step"] == 1000
    assert manifest["selection"]["selection_visible_reference"]["products"] == 82_264
    assert (
        manifest["selection"]["selection_visible_reference"]["heldout_products_excluded"] == 30_122
    )
    metrics = manifest["fresh_selection_metrics"]
    assert metrics["valid_products"] == metrics["all_three_components_reconstructed"]
    assert metrics["valid_products"] == metrics["exact_forward_reconstructions"]
    assert metrics["valid_products"] == metrics["all_three_handles_qualified"]
    assert metrics["usable_open_ended_successes"] == 847
    assert metrics["attempted_draws"] == 1024
    excluded = set(manifest["scope"]["excluded"])
    assert "L2 upstream-route proposal or route completion" in excluded
    assert "synthesis-value or route-value guidance" in excluded
    assert "biological-oracle guidance" in excluded


def test_v2_production_generator_manifest_authenticates_all_pinned_artifacts() -> None:
    manifest = json.loads(MANIFEST_V2.read_text())
    specifications = _artifact_specifications(manifest)
    assert len(specifications) >= 20
    for specification in specifications:
        path = REPO / specification["path"]
        assert path.is_file(), specification["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == specification["sha256"]


def test_v2_production_generator_manifest_freezes_confirmed_decoder_policy() -> None:
    manifest = json.loads(MANIFEST_V2.read_text())
    assert manifest["status"] == "frozen_after_independent_decoder_confirmation"
    assert manifest["supersession"]["base_manifest_rewritten"] is False
    assert manifest["identity"]["architecture"] == "full_morphology_program_conditioning"
    assert manifest["identity"]["checkpoint_step"] == 2000
    assert manifest["identity"]["terminal_decoder"] == "bond_stochastic"
    assert manifest["qualification"]["matched_design"]["matched_topology_rows_verified"] is True
    assert manifest["qualification"]["matched_design"]["only_terminal_bond_decoder_changed"] is True
    metrics = manifest["independent_confirmation_metrics"]
    assert metrics["valid_products"] == 987
    assert metrics["unique_valid_products"] == 977
    assert metrics["exact_l1_fraction_of_valid"] == 1.0
    assert metrics["all_confirmation_gates_pass"] is True
    assert (
        metrics["aldehyde_carbon_carbon_double_bond_fraction"]
        > metrics["matched_argmax_aldehyde_carbon_carbon_double_bond_fraction"]
    )
    assert manifest["next_draw_policy"]["fresh_program_draw_required"] is True
    assert manifest["next_draw_policy"]["confirmation_programs_are_evaluation_only"] is True
    excluded = set(manifest["scope"]["excluded"])
    assert "reuse of confirmation molecules for prospective candidate selection" in excluded
    assert (
        "production synthesis-value guidance before the existing dossier and zero-guidance gates pass"
        in excluded
    )


def test_v3_production_generator_manifest_authenticates_all_pinned_artifacts() -> None:
    manifest = json.loads(MANIFEST_V3.read_text())
    specifications = _artifact_specifications(manifest)
    assert len(specifications) == 18
    for specification in specifications:
        path = REPO / specification["path"]
        assert path.is_file(), specification["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == specification["sha256"]


def test_v3_production_generator_manifest_freezes_refit_selection() -> None:
    manifest = json.loads(MANIFEST_V3.read_text())
    assert manifest["status"] == "frozen_after_all_fold_refit_checkpoint_selection"
    assert manifest["supersession"]["base_manifest_rewritten"] is False
    assert manifest["identity"]["checkpoint_step"] == 1000
    assert manifest["identity"]["terminal_decoder"] == "bond_stochastic"
    assert manifest["identity"]["maximum_adjacent_branch_runs_by_role"] == [2, 1, 1]
    assert manifest["frozen_inputs"]["all_fold_morphology_program_prior"]["role_state_counts"] == [
        106,
        39,
        40,
    ]
    metrics = manifest["independent_selection_metrics"]
    assert metrics["attempted_draws"] == 1024
    assert metrics["valid_products"] == 994
    assert metrics["unique_valid_products"] == 993
    assert metrics["usable_open_ended_successes"] == 938
    assert metrics["all_selection_gates_pass"] is True
    assert manifest["next_draw_policy"]["fresh_program_draw_required"] is True
    assert manifest["next_draw_policy"]["checkpoint_selection_programs_are_evaluation_only"] is True
