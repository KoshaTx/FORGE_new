from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from forge.value.synthesis import ComponentSynthesisValue, ProductSynthesisValue
from forge.value.ugi3_synthesis_value_audit import (
    Ugi3SynthesisValueAuditError,
    build_ugi3_synthesis_value_audit,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts/phase1_build_ugi3_synthesis_values.py"
SPEC = importlib.util.spec_from_file_location("phase1_build_ugi3_synthesis_values", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)
CONFIG = REPO / "configs/route/phase1_ugi3_synthesis_value_audit_v1.json"


def _build(config: Path = CONFIG):
    return build_ugi3_synthesis_value_audit(
        config_path=config,
        input_paths=SCRIPT.INPUT_PATHS,
        targeted_audit_input_paths=SCRIPT.TARGETED_AUDIT_INPUT_PATHS,
        role_gap_input_paths=SCRIPT.ROLE_GAP_INPUT_PATHS,
        head_terminal_input_paths=SCRIPT.HEAD_TERMINAL_INPUT_PATHS,
    )


def _load_payload(payload: bytes):
    return json.loads(gzip.decompress(payload))


def test_synthesis_value_audit_is_reproducible_and_nonprobabilistic() -> None:
    first = _build()
    second = _build()

    assert first == second
    result, component_ledger, product_ledger = first
    assert result["summary"]["registry_outcomes"] == {
        "complete": 48,
        "missing_knowledge": 273,
        "outside_support": 103,
    }
    assert result["summary"]["generated_product_outcomes"] == {
        "complete": 38,
        "noncomplete": 969,
    }
    assert result["summary"]["non_null_scalar_values"] == 0
    assert result["summary"]["non_null_success_probabilities"] == 0

    components = _load_payload(component_ledger)["records"]
    products = _load_payload(product_ledger)["records"]
    assert len(components) == 610
    assert len(products) == 1007
    assert all(ComponentSynthesisValue.from_dict(row["value"]) for row in components)
    assert all(ProductSynthesisValue.from_dict(row["value"]) for row in products)
    assert all(row["value"]["scalar_value"] is None for row in products)
    assert all(row["value"]["success_probability"] is None for row in products)


def test_generated_catalog_absence_remains_missing_knowledge() -> None:
    _, component_ledger, _ = _build()
    records = _load_payload(component_ledger)["records"]
    absent = [
        row
        for row in records
        if row["source_class"] == "generated_component_missing_route_knowledge"
    ]

    assert len(absent) == 519
    assert all(row["value"]["assessment_outcome"] == "missing_knowledge" for row in absent)
    assert all(row["value"]["missing_knowledge_leaf_count"] == 1 for row in absent)
    assert all(row["value"]["incompatible_leaf_count"] == 0 for row in absent)


def test_frozen_summary_detects_semantic_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["expected_summary"]["generated_product_outcomes"]["complete"] = 39
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    with pytest.raises(Ugi3SynthesisValueAuditError, match="summary changed"):
        _build(config_path)
