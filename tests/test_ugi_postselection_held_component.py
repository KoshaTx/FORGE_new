from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forge.design.corpus.ugi_postselection_held_component import (
    component_membership_class,
    evaluate_postselection_held_component_stress,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_product_l1_postselection_held_component_stress_v1.json"
RESULT = REPO / "results/phase1/ugi_product_l1_postselection_held_component_stress_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_component_membership_class_uses_mutually_exclusive_precedence() -> None:
    catalog = {
        "amine_head": {"CN": "heldout"},
        "oxoester_aldehyde_body_tail": {"CC=O": "calibration"},
        "isocyanide_tail": {"[C-]#[N+]C": "train"},
    }
    components = {
        "amine_head": "CN",
        "oxoester_aldehyde_body_tail": "CC=O",
        "isocyanide_tail": "[C-]#[N+]C",
    }
    assert component_membership_class(components, catalog) == "exact_heldout_component"
    components["amine_head"] = "CCN"
    assert component_membership_class(components, catalog) == "genuinely_generated_component"


def test_frozen_postselection_result_has_explicit_nonpristine_claim_boundary() -> None:
    result = json.loads(RESULT.read_text())
    assert result["status"] == "complete_descriptive_postselection_evaluation"
    assert result["evaluation_timing"]["heldout_fold_is_pristine"] is False
    assert result["selected_checkpoint"]["step"] == 1000
    assert result["selected_checkpoint"]["selection_remains_frozen"] is True
    assert result["decision"] == {
        "architecture_or_checkpoint_changed": False,
        "broad_pretraining_decision": "not_made_by_this_descriptive_evaluation",
        "thresholds_applied": False,
    }
    overall = result["generation"]["overall"]
    assert overall["attempted_programs"] == 512
    assert overall["valid_products"]["count"] == 511
    assert overall["exact_forward_reconstruction"]["count"] == 511
    assert overall["usable_structural_open_ended"]["count"] == 429
    assert overall["product_membership_counts"] == {
        "heldout": 5,
        "outside_frozen_product_corpus": 488,
        "train": 18,
    }
    assert result["single_corruption_denoising_loss"]["records"] == 30_122
    assert len(result["generation"]["by_held_role_class"]) == 7


def test_postselection_evaluation_recomputes_deterministically(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    observed = evaluate_postselection_held_component_stress(
        config_path=CONFIG,
        output_path=output,
        repository=REPO,
    )
    assert observed == json.loads(RESULT.read_text())
    assert json.loads(output.read_text()) == observed
    for specification in observed["inputs"].values():
        assert _sha256(REPO / specification["path"]) == specification["sha256"]
