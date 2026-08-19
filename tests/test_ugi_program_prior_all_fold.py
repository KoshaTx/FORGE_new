from __future__ import annotations

import importlib.util
from pathlib import Path

from forge.product.ugi_program_prior import load_program_prior

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/phase1_build_ugi_program_prior_all_fold.py"
SPEC = importlib.util.spec_from_file_location("phase1_build_ugi_program_prior_all_fold", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_all_fold_prior_uses_every_frozen_production_record() -> None:
    result = MODULE.build_all_fold_prior(
        REPO / "configs/model/phase1_ugi_joint_sparse_balanced_v2_production_refit.json"
    )
    assert result["population"] == "all_fold_production_refit"
    assert result["fold_counts"] == {
        "train": 66464,
        "calibration": 15800,
        "heldout": 30122,
    }
    assert result["products"] == 112386
    assert result["support_audit"]["support_counts_by_role"] == {
        "amine_head": 106,
        "oxoester_aldehyde_body_tail": 39,
        "isocyanide_tail": 40,
    }


def test_all_fold_prior_strictly_extends_development_support() -> None:
    result = MODULE.build_all_fold_prior(
        REPO / "configs/model/phase1_ugi_joint_sparse_balanced_v2_production_refit.json"
    )
    development, _ = load_program_prior(REPO / "results/phase1/ugi_program_prior_v2.json")
    for all_records, development_support in zip(
        result["role_priors"].values(), development.support_by_role, strict=True
    ):
        all_support = {
            (
                int(record["node_count"]),
                int(record["junction_budget"]),
                int(record["cycle_rank"]),
                int(record["attachment_count"]),
            )
            for record in all_records
        }
        assert set(development_support) < all_support
