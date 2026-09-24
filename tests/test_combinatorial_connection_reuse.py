from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.phase1.multireaction import combinatorial_connection_reuse as experiment
from forge.core.hashing import sha256_file
from forge.model.precursor_reuse import PrecursorReuseError


@pytest.fixture
def saved(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    for name in experiment.SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / name).read_bytes())
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}\n")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": experiment.SCHEMA + "_config",
                "policy": experiment.POLICY,
                "baseline_result": experiment._pin(baseline, tmp_path),
            }
        )
    )
    expected = (
        {"total_exact_gain": 1, "all_family_preservation_screen_passed": True},
        {"attempts.jsonl": [{"sample_index": 0, "disposition": "original_invalid"}]},
    )
    # Isolate authentication and semantic comparison. Real chemistry is replayed by the CLI.
    monkeypatch.setattr(experiment, "calculate", lambda *_: copy.deepcopy(expected))
    experiment.run(tmp_path, config, tmp_path / "output")
    return tmp_path, tmp_path / "output/result.json"


def test_fresh_output_and_semantic_verification(saved):
    repo, path = saved
    assert experiment.verify(repo, path)["attempts_recomputed"] == 1
    with pytest.raises(PrecursorReuseError, match="fresh"):
        experiment.run(repo, repo / "config.json", path.parent)


@pytest.mark.parametrize("change", ["summary", "ledger", "baseline", "policy"])
def test_verifier_rejects_substitution_even_with_updated_artifact_hash(saved, change):
    repo, path = saved
    result = json.loads(path.read_text())
    if change == "summary":
        result["total_exact_gain"] += 1
    elif change == "ledger":
        pin = result["artifacts"]["attempts.jsonl"]
        ledger = repo / pin["path"]
        ledger.write_text('{"sample_index":0,"disposition":"promoted"}\n')
        pin["sha256"] = str(sha256_file(ledger))
    elif change == "baseline":
        other = repo / "other.json"
        other.write_text("{}\n")
        result["baseline_result"] = experiment._pin(other, repo)
    else:
        result["policy"]["source_program_connections_used"] = False
    path.write_text(json.dumps(result))
    with pytest.raises(PrecursorReuseError):
        experiment.verify(repo, path)
