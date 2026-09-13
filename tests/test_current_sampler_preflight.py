import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.phase1.synthesis_guidance import current_sampler_preflight as preflight
from forge.core.hashing import sha256_file


def test_missing_evidence_is_an_infrastructure_failure_not_a_candidate_score(tmp_path, monkeypatch):
    config = tmp_path / "source.json"
    config.write_text(
        json.dumps({"inputs": {"ledger": {"path": "missing.json", "sha256": "a" * 64}}})
    )
    historical = tmp_path / "historical.json"
    historical.write_text(
        json.dumps(
            {
                "assessment_as_of_utc": "2026-08-03T17:30:13Z",
                "cumulative_source_inputs_sha256": "b" * 64,
            }
        )
    )
    monkeypatch.setattr(
        preflight,
        "source_qualified_cumulative_paths",
        lambda *_: SimpleNamespace(ledger=tmp_path / "missing.json"),
    )
    calls = []

    def unavailable(**kwargs):
        calls.append(kwargs)
        raise FileNotFoundError("exact evidence ledger missing")

    monkeypatch.setattr(preflight, "load_cumulative_production_ugi3_source", unavailable)
    before = sha256_file(config)
    result = preflight.inspect_route_source(tmp_path, config, historical)
    assert result["status"] == "source_authentication_failed"
    assert result["pinned_inputs"]["ledger"]["status"] == "unavailable"
    assert result["candidate_assessment_calls"] == 0
    assert result["errors"][0]["type"] == "FileNotFoundError"
    assert len(calls) == 1
    assert calls[0]["assessment_as_of_utc"] == "2026-08-03T17:30:13Z"
    assert "route_values" not in result
    assert sha256_file(config) == before


def test_nonzero_config_rejected_before_sampler_or_source_access(tmp_path):
    config = tmp_path / "config.json"
    policy = {**preflight.POLICY, "guidance_strength": 0.25}
    config.write_text(
        json.dumps(
            {"schema_version": preflight.SCHEMA, "runtime": preflight.RUNTIME, "policy": policy}
        )
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="zero-only policy"):
        preflight.run_preflight(tmp_path, config, output)
    assert not output.exists()


def test_preflight_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("original run")
    with pytest.raises(ValueError, match="fresh output"):
        preflight.run_preflight(tmp_path, Path("missing_config"), output)
    assert sentinel.read_text() == "original run"
