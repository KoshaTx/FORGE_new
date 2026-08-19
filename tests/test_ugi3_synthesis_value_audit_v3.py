from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from forge.value.synthesis import ComponentSynthesisValue, ProductSynthesisValue
from forge.value.ugi3_synthesis_value_audit_v3 import (
    Ugi3SynthesisValueAuditV3Error,
    build_ugi3_synthesis_value_audit_v3,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts/phase1_build_ugi3_synthesis_values_v3.py"
SPEC = importlib.util.spec_from_file_location("phase1_build_ugi3_synthesis_values_v3", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)
CONFIG = REPO / "configs/route/phase1_ugi3_synthesis_value_audit_v3.json"


def _build(config: Path = CONFIG):
    return build_ugi3_synthesis_value_audit_v3(
        config_path=config,
        input_paths=SCRIPT.INPUT_PATHS,
        third_wave_input_paths=SCRIPT.THIRD_WAVE_INPUT_PATHS,
    )


def _load_payload(payload: bytes):
    return json.loads(gzip.decompress(payload))


def test_v3_is_reproducible_and_keeps_v2_immutable() -> None:
    first = _build()
    second = _build()

    assert first == second
    result, component_ledger, product_ledger = first
    assert result["summary"]["third_wave_components_changed"] == 1
    assert result["summary"]["generated_component_outcomes"] == {
        "complete": 26,
        "missing_knowledge": 560,
        "outside_support": 24,
    }
    assert result["summary"]["generated_product_outcomes"] == {
        "complete": 45,
        "noncomplete": 962,
    }
    assert result["summary"]["non_null_scalar_values"] == 0
    assert result["summary"]["non_null_success_probabilities"] == 0

    components = _load_payload(component_ledger)["records"]
    products = _load_payload(product_ledger)["records"]
    assert len(components) == 610
    assert len(products) == 1007
    assert all(ComponentSynthesisValue.from_dict(row["value"]) for row in components)
    assert all(ProductSynthesisValue.from_dict(row["value"]) for row in products)


def test_v3_changes_only_exact_cyclohexylamine_component() -> None:
    _, component_ledger, _ = _build()
    records = _load_payload(component_ledger)["records"]
    changed = [row for row in records if row["source_class"] == "third_wave_exact_current_terminal"]

    assert len(changed) == 1
    assert changed[0]["role"] == "amine_head"
    assert changed[0]["canonical_smiles"] == "NC1CCCCC1"
    assert changed[0]["value"]["assessment_outcome"] == "complete"
    assert changed[0]["value"]["protection_burden"]["count"] is None
    assert changed[0]["value"]["purification_burden"]["count"] is None


def test_v3_summary_detects_semantic_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["expected_summary"]["generated_product_outcomes"]["complete"] = 46
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    with pytest.raises(Ugi3SynthesisValueAuditV3Error, match="summary changed"):
        _build(config_path)
