from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.archive.phase1.synthesis_value_audits.synthesis_source_supersession import (
    AUTHORIZED_REPLAY_IDS,
    SynthesisSourceSupersessionError,
    build_synthesis_source_supersession_audit,
    substitute_value_source_hash,
)
from forge.corpus.r1_prime_audit import sha256_file

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_synthesis_source_supersession_audit_v1.json"
OLD_SHA256 = "dba5b11067513e4f431f78f3b9a316dfe0d58c5f046210c170c55b9a0e241e3c"
NEW_SHA256 = "6e77063f45012144912883f16a12698eda809446635c94d59d1ab6211993f189"


def test_substitution_changes_only_asset_style_source_hash() -> None:
    original = {
        "schema_version": "example.v1",
        "inputs": {
            "value_source": {
                "asset": "src/forge/value/synthesis.py",
                "expected_sha256": OLD_SHA256,
            },
            "other": {"asset": "other", "expected_sha256": "unchanged"},
        },
    }
    frozen = copy.deepcopy(original)
    replay, substitution = substitute_value_source_hash(
        original,
        old_sha256=OLD_SHA256,
        new_sha256=NEW_SHA256,
    )

    assert original == frozen
    assert replay["inputs"]["value_source"]["expected_sha256"] == NEW_SHA256
    assert replay["inputs"]["value_source"]["asset"] == "src/forge/value/synthesis.py"
    assert replay["inputs"]["other"] == original["inputs"]["other"]
    assert substitution == {
        "json_path": "inputs.value_source.expected_sha256",
        "expected_before": OLD_SHA256,
        "observed_after": NEW_SHA256,
    }


def test_substitution_rejects_source_record_scope_expansion() -> None:
    malformed = {
        "inputs": {
            "value_source": {
                "path": "src/forge/value/synthesis.py",
                "sha256": OLD_SHA256,
                "allow_drift": True,
            }
        }
    }
    with pytest.raises(SynthesisSourceSupersessionError, match="record shape changed"):
        substitute_value_source_hash(
            malformed,
            old_sha256=OLD_SHA256,
            new_sha256=NEW_SHA256,
        )


def test_full_replay_is_behavior_preserving_and_does_not_rewrite_history() -> None:
    config = json.loads(CONFIG.read_text())
    historical_paths = []
    for replay in config["replays"].values():
        historical_paths.extend(REPO / record["path"] for record in replay.values())
    before = {path: sha256_file(path) for path in historical_paths}

    result = build_synthesis_source_supersession_audit(REPO, CONFIG)

    assert result["status"] == "behavior_preserving_source_supersession_qualified"
    assert result["summary"] == {
        "replay_targets": 4,
        "synthesis_value_replays": 3,
        "fresh_pool_route_coverage_replays": 1,
        "byte_identical_component_ledgers": 4,
        "byte_identical_product_ledgers": 4,
        "semantically_identical_summaries": 4,
        "semantically_identical_normalized_results": 4,
        "behavior_preserving": True,
    }
    assert tuple(result["replays"]) == AUTHORIZED_REPLAY_IDS
    assert all(replay["behavior_preserving"] for replay in result["replays"].values())
    assert all(
        replay["raw_result"]["byte_identical"] is False for replay in result["replays"].values()
    )
    assert result["adjudication"] == {
        "historical_configs_rewritten": False,
        "historical_ledgers_rewritten": False,
        "production_pin_update_authorized": False,
        "vnext_qualification_chain_required": True,
        "sealed_holdout_accessed": False,
        "synthesis_guidance_authorized": False,
    }
    assert {path: sha256_file(path) for path in historical_paths} == before


def test_candidate_source_hash_remains_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    config["source_supersession"]["candidate_sha256"] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config))

    with pytest.raises(
        SynthesisSourceSupersessionError, match="candidate value source hash changed"
    ):
        build_synthesis_source_supersession_audit(REPO, changed)
