from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.product.ugi_selected_guidance_adapter import (
    _particle_seed_manifest_sha256,
)
from forge.product.ugi_selected_v2_pool_singleton_equivalence_v2 import (
    EXPECTED_BASE_ADAPTER_IDENTITY_SHA256,
    EXPECTED_IMPLEMENTATION_SHA256,
    EXPECTED_INPUT_KEYS,
    EXPECTED_RUNTIME_VERSIONS,
    UgiSelectedV2PoolSingletonEquivalenceV2Error,
    _validate_config,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_selected_v2_pool_singleton_equivalence_v2.json"


def test_v2_equivalence_config_binds_qualified_runtime_and_prior_chain() -> None:
    config, paths, prior, qualification = _validate_config(REPO, CONFIG)

    assert frozenset(paths) == EXPECTED_INPUT_KEYS
    assert qualification["implementation_sha256"] == EXPECTED_IMPLEMENTATION_SHA256
    assert tuple(tuple(item) for item in qualification["runtime_versions"]) == (
        EXPECTED_RUNTIME_VERSIONS
    )
    assert prior["selected_generator"]["adapter_identity_sha256"] == (
        EXPECTED_BASE_ADAPTER_IDENTITY_SHA256
    )
    assert config["scope"]["nonzero_guidance"] is False


def test_v2_equivalence_config_rejects_input_schema_drift(tmp_path: Path) -> None:
    changed = json.loads(CONFIG.read_text())
    changed["inputs"]["decorative_unvalidated_input"] = {
        "path": "pyproject.toml",
        "sha256": "0" * 64,
    }
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed))

    with pytest.raises(
        UgiSelectedV2PoolSingletonEquivalenceV2Error,
        match="input set changed",
    ):
        _validate_config(REPO, changed_path)


def test_particle_seed_manifests_are_order_and_membership_sensitive() -> None:
    canonical = _particle_seed_manifest_sha256((11, 22, 33))

    assert canonical != _particle_seed_manifest_sha256((33, 22, 11))
    assert canonical != _particle_seed_manifest_sha256((11, 22))
    assert canonical != _particle_seed_manifest_sha256((11, 22, 34))
