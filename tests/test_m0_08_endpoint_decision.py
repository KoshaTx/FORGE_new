from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.bio.endpoint_decision import (
    EndpointDecisionError,
    freeze_endpoint_decision,
)

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/bio/m0_08_endpoint_decision.json"


def test_endpoint_decision_freezes_without_an_implicit_default(tmp_path: Path) -> None:
    output = tmp_path / "result.json"

    result = freeze_endpoint_decision(CONFIG, output, REPO)

    assert result["status"] == "decision_package_complete_endpoint_unlocked"
    assert result["decision"]["endpoint_locked"] is False
    assert result["decision"]["default_in_code"] is None
    assert set(result["endpoint_specifications"]) == {
        "im_functional_editing",
        "im_vaccination",
        "liver_functional_editing",
    }
    assert json.loads(output.read_text()) == result


def test_endpoint_decision_fails_closed_on_input_hash_change(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["inputs"]["decision_package"]["sha256"] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config))

    with pytest.raises(EndpointDecisionError, match="hash mismatch"):
        freeze_endpoint_decision(changed, tmp_path / "result.json", REPO)
