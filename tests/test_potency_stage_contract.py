"""Exercise stage dispatch without fitting models or submitting computation."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

from experiments.phase1.hela_potency import potency_adapter, stages


def test_crossfit_stage_obeys_the_trainer_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = create_autospec(
        potency_adapter.run_potency_adapter_crossfit,
        return_value={
            "status": "pass",
            "observations": 12,
            "folds": [{}, {}],
            "active_time_bins": [0.25],
        },
    )
    monkeypatch.setattr(potency_adapter, "run_potency_adapter_crossfit", trainer)
    context = SimpleNamespace(
        config=lambda: {"inputs": {}},
        config_path=tmp_path / "config.json",
        repo=tmp_path,
        output_path=lambda _: tmp_path / "output",
        profile="smoke",
        resources=SimpleNamespace(device="cpu"),
        inputs={},
        stage=SimpleNamespace(stage_id="unit-fixture"),
    )

    result = stages.fit_ugi_potency_adapter_crossfit(context)

    trainer.assert_called_once_with(
        context.config_path,
        tmp_path,
        tmp_path / "output",
        profile="smoke",
        allocated_device="cpu",
    )
    assert result.metrics == {"observations": 12, "crossfit_folds": 2, "active_time_bins": 1}
    assert {artifact.label for artifact in result.artifacts} == {"result", "adapter_bundle"}
    assert result.summary["guided_generation"] is False
    assert not (tmp_path / "output").exists()
