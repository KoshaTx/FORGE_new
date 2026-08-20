from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from forge.corpus.source_platform_registry_audit import (
    RESULT_SCHEMA_VERSION,
    SourcePlatformAuditError,
    _classify_platform,
    _load_platform_specs,
    load_config,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/corpus/m0_04_source_platform_registry_audit.json"
RESULT = REPO / "results/m0_04/source_platform_registry_audit.json"


def test_platform_classifier_requires_exactly_one_declared_cohort() -> None:
    specs = _load_platform_specs(
        [
            {
                "platform_id": "one",
                "aliases": ["ONE"],
                "expected_structures": 1,
                "reported_final_assembly": {
                    "status": "verified",
                    "description": "first",
                },
            },
            {
                "platform_id": "two",
                "aliases": ["TWO"],
                "expected_structures": 1,
                "reported_final_assembly": {
                    "status": "verified",
                    "description": "second",
                },
            },
        ]
    )
    row = {
        "r0_structure_id": "R0-test",
        "study_split_groups_json": '{"source": ["ONE"]}',
    }
    assert _classify_platform(row, specs) == "one"
    row["study_split_groups_json"] = '{"source": ["ONE", "TWO"]}'
    with pytest.raises(SourcePlatformAuditError, match="expected one platform match"):
        _classify_platform(row, specs)


def test_config_freezes_complete_source_holdout() -> None:
    config, specs = load_config(CONFIG)
    assert config["expected_source_study_heldout"] == 2_333
    assert sum(spec.expected_structures for spec in specs) == 2_333
    assert {spec.platform_id for spec in specs} == {
        "lm_2019_lm_3cr",
        "jl_2024",
        "xh_2025",
        "sx_2025",
    }
    assert config["claims_boundary"]["not_a_model_generalization_benchmark"] is True


def test_completed_source_platform_artifacts_match_manifest() -> None:
    if not RESULT.exists():
        pytest.skip("source-platform audit has not been generated")
    result = json.loads(RESULT.read_text())
    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["summary"]["source_study_heldout"] == 2_333
    assert result["summary"]["products_with_fully_training_pool_supported_decomposition"] == 0
    assert result["decision"]["heldout_components_added_to_pool"] is False
    assert result["decision"]["source_holdout_is_model_generalization_benchmark"] is False
    for details in result["artifacts"].values():
        path = REPO / details["path"]
        payload = path.read_bytes()
        assert len(payload) == details["bytes"]
        assert hashlib.sha256(payload).hexdigest() == details["sha256"]
        with gzip.open(path, "rt") as handle:
            assert sum(1 for _ in handle) == 2_334
