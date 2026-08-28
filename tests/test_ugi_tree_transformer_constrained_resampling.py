from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments._runtime.spec import ExperimentSpec
from experiments.catalog import SPECIFICATIONS
from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_constrained_resampling import (  # noqa: E501
    UgiTreeTransformerConstrainedResamplingError,
    _runtime,
    run_tree_transformer_constrained_resampling,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO
    / "configs/model/phase1_ugi_tree_relational_constrained_resampling_v1.json"
)
CONFIG_V2 = (
    REPO
    / "configs/model/phase1_ugi_tree_relational_constrained_resampling_v2.json"
)
H100_SPEC = (
    REPO
    / "experiments/phase1/product_l1/"
    "ugi_tree_relational_edge_constrained_resampling_seed0_h100_v1.json"
)


def test_frozen_constrained_resampling_profiles_are_paired() -> None:
    config = json.loads(CONFIG.read_text())

    smoke = _runtime(config, "smoke")
    full = _runtime(config, "full")

    assert smoke["program_count"] == 32
    assert full["program_count"] == 3072
    assert smoke["terminal_decoder_mode"] == full["terminal_decoder_mode"] == "stochastic"
    assert config["policy"]["retraining"] is False
    assert config["policy"]["repairs_or_retries"] is False
    assert config["policy"]["component_identities_exposed_to_decoder"] is False


def test_edge_scope_followup_adds_paired_noninferiority_gates() -> None:
    config = json.loads(CONFIG_V2.read_text())

    _runtime(config, "smoke")

    assert config["policy"]["local_chemistry_constraint_scope"] == "role_edges_only"
    assert (
        config["gates"][
            "minimum_paired_original_local_supported_exact_l1_retained_fraction"
        ]
        == 1.0
    )
    assert "broad_scope_smoke" in config["inputs"]


def test_full_edge_constrained_diagnostic_is_one_exact_h100_inference_run() -> None:
    config = json.loads(CONFIG_V2.read_text())
    spec = ExperimentSpec.load(H100_SPEC)
    stage = spec.stages[0]

    assert SPECIFICATIONS[spec.experiment_id] == str(H100_SPEC.relative_to(REPO))
    assert spec.profiles == ("full",)
    assert spec.replicates == {"full": 1}
    assert len(spec.stages) == 1
    assert stage.implementation == (
        "evaluate.ugi.tree-relational-edge-constrained-resampling.v1"
    )
    assert stage.resources.device == "cuda"
    assert stage.resources.gpu_type == "H100!"
    assert {label: pin.to_mapping() for label, pin in stage.inputs.items()} == config[
        "inputs"
    ]
    assert config["profiles"]["full"]["program_count"] == 3072
    assert config["policy"]["retraining"] is False
    assert config["policy"]["repairs_or_retries"] is False


def test_constrained_resampling_rejects_unknown_profile() -> None:
    with pytest.raises(UgiTreeTransformerConstrainedResamplingError):
        _runtime(json.loads(CONFIG.read_text()), "development")


def test_constrained_resampling_does_not_overwrite_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(UgiTreeTransformerConstrainedResamplingError, match="already exists"):
        run_tree_transformer_constrained_resampling(
            CONFIG,
            REPO,
            output,
            profile="smoke",
            device="cpu",
        )
