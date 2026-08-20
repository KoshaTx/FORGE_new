from __future__ import annotations

import json

from forge.design.corpus.ugi_expanded_enumeration import (
    PRODUCT_FIELDS,
    _component_novelty,
    _product_fold,
    _product_record,
)


def _component(role: str, fold: str, current: bool, classes: list[str]) -> dict[str, str]:
    return {
        "role": role,
        "canonical_smiles": {
            "amine_head": "CN",
            "oxoester_aldehyde_body_tail": "CC=O",
            "isocyanide_tail": "[C-]#[N+]C",
        }[role],
        "family_id": f"{role}-family",
        "family_fold": fold,
        "family_size": "2",
        "is_current_catalog": str(current).lower(),
        "source_classes_json": json.dumps(classes),
    }


def test_product_fold_is_strict_across_roles() -> None:
    assert _product_fold(["train", "train", "train"]) == "train"
    assert _product_fold(["train", "calibration", "train"]) == "calibration"
    assert _product_fold(["heldout", "train", "calibration"]) == "heldout"


def test_component_novelty_distinguishes_transferred_support() -> None:
    familiar = [
        _component(role, "train", True, ["current_phase1_catalog"])
        for role in ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
    ]
    assert _component_novelty(familiar) == "familiar_components"
    transferred = list(familiar)
    transferred[1] = _component(
        "oxoester_aldehyde_body_tail",
        "train",
        False,
        ["cross_platform_hydrophobic_transfer"],
    )
    assert _component_novelty(transferred) == "transferred_known_component"


def test_product_record_uses_inverse_family_size_weight() -> None:
    components = [
        _component("amine_head", "train", True, ["current_phase1_catalog"]),
        _component("oxoester_aldehyde_body_tail", "train", True, ["current_phase1_catalog"]),
        _component("isocyanide_tail", "train", True, ["current_phase1_catalog"]),
    ]
    record = _product_record("CC", components, source_stratum="test", measured=False)
    values = dict(zip(PRODUCT_FIELDS, record, strict=True))
    assert values["primary_product_fold"] == "train"
    assert float(values["family_balance_weight_raw"]) == 0.125
    assert values["component_novelty_class"] == "familiar_components"
