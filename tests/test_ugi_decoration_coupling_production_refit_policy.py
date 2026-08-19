from __future__ import annotations

import hashlib
import json
from pathlib import Path

from forge.product.ugi_joint_sparse_training import _training_partition

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/model/phase1_ugi_decoration_coupling_production_refit_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_refit_contract_preserves_selected_exposure() -> None:
    config = json.loads(CONFIG.read_text())
    duration = config["duration_contract"]
    expected = (
        round(
            duration["selected_development_checkpoint_step"]
            * duration["production_training_records"]
            / duration["development_training_records"]
            / 100
        )
        * 100
    )

    assert duration["selected_development_checkpoint_step"] == 3000
    assert duration["fixed_production_steps"] == expected == 5100
    assert config["full"]["steps"] == 5100
    assert config["full"]["early_stopping"]["patience"] == 0
    assert config["model"]["decoration_state_conditioning"] == "bidirectional_anchor_local"


def test_production_refit_uses_all_folds_without_selection_loss() -> None:
    config = json.loads(CONFIG.read_text())
    partition = _training_partition(
        config,
        tuple(config["expected_fold_counts"]),
        config["full"],
    )

    assert set(partition.training_folds) == {"train", "calibration", "heldout"}
    assert partition.selection_mode == "fixed_final_step"
    assert set(partition.overlapping_folds) == {"train", "calibration", "heldout"}


def test_production_refit_selection_and_data_hashes_are_frozen() -> None:
    config = json.loads(CONFIG.read_text())
    records = [
        *config["inputs"].values(),
        *config["selection_evidence"].values(),
    ]

    for record in records:
        path = REPO / record["path"]
        assert path.is_file()
        assert _sha256(path) == record["sha256"]

    confirmation = config["selection_evidence"]["independent_confirmation"]
    result = json.loads((REPO / confirmation["path"]).read_text())
    assert result["decision"]["selected_checkpoint_step"] == 3000
