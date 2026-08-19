from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from forge.bio.oracle_lantern import (
    LanternReproductionError,
    load_lantern_config,
    load_numpy_feature_dictionary,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bio/m0_07_lantern_reproduction.json"
RESULT = ROOT / "results/m0_07/lantern_reproduction_result.json"


def test_config_freezes_diagnostic_only_policy() -> None:
    config = load_lantern_config(CONFIG)
    assert config["policy"]["selection_eligible"] is False
    assert config["policy"]["oracle_model_frozen"] is False
    assert config["preprocessing"]["leakage_status"] == "test_information_leakage"
    assert config["randomness"]["training_performed"] is False


def test_allowlisted_numpy_feature_pickle(tmp_path: Path) -> None:
    path = tmp_path / "features.pkl"
    path.write_bytes(pickle.dumps({"CC": np.asarray([1.0, 2.0])}, protocol=4))
    loaded = load_numpy_feature_dictionary(
        path,
        expected_rows=1,
        expected_columns=2,
        label="test features",
    )
    np.testing.assert_array_equal(loaded["CC"], np.asarray([1.0, 2.0]))


def test_non_numpy_pickle_global_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.pkl"
    path.write_bytes(pickle.dumps(eval, protocol=4))
    with pytest.raises(LanternReproductionError, match="prohibited pickle global"):
        load_numpy_feature_dictionary(
            path,
            expected_rows=1,
            expected_columns=1,
            label="unsafe features",
        )


def test_reproduction_artifact_contract() -> None:
    result = json.loads(RESULT.read_text())
    assert result["schema_version"] == "m0_07_lantern_reproduction.v1"
    assert result["status"] == "released_checkpoint_reproduced_diagnostic_only"
    assert result["policy"]["selection_eligible"] is False
    assert result["policy"]["calibration_claim_allowed"] is False
    assert result["metrics"]["test"]["rows"] == 110
    assert result["reproduction_contract"]["feature_dimensions"] == {
        "circular": 2048,
        "expert": 210,
        "combined": 2258,
    }

    for name, record in result["inputs"].items():
        path = ROOT / record["path"]
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        assert path.stat().st_size == record["bytes"]

    prediction_record = result["artifact"]
    prediction_path = ROOT / prediction_record["path"]
    payload = prediction_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == prediction_record["sha256"]
    assert len(payload) == prediction_record["bytes"]
    with gzip.open(prediction_path, "rt") as handle:
        assert sum(1 for _ in handle) == 1101
