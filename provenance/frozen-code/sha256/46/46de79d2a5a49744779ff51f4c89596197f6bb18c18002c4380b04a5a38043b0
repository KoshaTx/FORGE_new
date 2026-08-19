from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Any

import pytest

from forge.data.r1_prime_audit import sha256_file
from forge.route.ugi3_frozen_parent_component_ledger import (
    DERIVED_RECORD_FIELDS,
    FROZEN_STATUS,
    LEDGER_SCHEMA_VERSION,
    REQUIRED_POLICY,
    Ugi3FrozenParentComponentLedgerError,
    build_frozen_parent_component_ledger,
)
from forge.route.ugi3_route_registry_pair_builder import _load_parent_records
from forge.route.ugi3_route_registry_pair_contract import TARGET_ROLE, TARGET_SMILES

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/route/phase1_ugi3_frozen_parent_component_ledger_v1.json"
RESULT = REPO / "results/phase1/ugi3_frozen_parent_component_ledger_v1/result.json"
LEDGER = REPO / (
    "results/phase1/ugi3_frozen_parent_component_ledger_v1/" "final_component_ledger.json.gz"
)
V5_LEDGER = REPO / (
    "results/phase1/ugi3_fresh_pool_route_coverage_v5/component_synthesis_values.json.gz"
)
V3_LEDGER = REPO / (
    "results/phase1/ugi3_fresh_pool_route_coverage_v3/component_synthesis_values.json.gz"
)


def _stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _gzip_bytes(value: Any) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(_stable_json_bytes(value))
    return output.getvalue()


def _gzip_payload(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _by_key(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(record["role"], record["canonical_smiles"]): record for record in records}


def _strip_parent_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in DERIVED_RECORD_FIELDS}


def _changed_config(tmp_path: Path, mutate) -> Path:
    config = json.loads(CONFIG.read_text())
    mutate(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return path


def test_parent_ledger_is_deterministic_stored_and_config_owned() -> None:
    first_result, first_ledger = build_frozen_parent_component_ledger(REPO, CONFIG)
    second_result, second_ledger = build_frozen_parent_component_ledger(REPO, CONFIG)
    assert first_result == second_result == json.loads(RESULT.read_text())
    assert first_ledger == second_ledger == LEDGER.read_bytes()
    assert first_result["status"] == FROZEN_STATUS
    assert first_result["config_sha256"] == sha256_file(CONFIG)
    assert first_result["artifacts"]["final_component_ledger.json.gz"] == {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "sha256": sha256_file(LEDGER),
    }
    payload = _gzip_payload(LEDGER)
    assert payload["schema_version"] == LEDGER_SCHEMA_VERSION
    assert payload["status"] == FROZEN_STATUS
    assert payload["config_sha256"] == sha256_file(CONFIG)


def test_exactly_one_authorized_c18_reversion_preserves_all_other_v5_records() -> None:
    v5_records = _gzip_payload(V5_LEDGER)["records"]
    v3_records = _gzip_payload(V3_LEDGER)["records"]
    output_records = _gzip_payload(LEDGER)["records"]
    assert len(v5_records) == len(v3_records) == len(output_records) == 1919
    v5_by_key = _by_key(v5_records)
    v3_by_key = _by_key(v3_records)
    output_by_key = _by_key(output_records)
    assert set(v5_by_key) == set(v3_by_key) == set(output_by_key)

    source_differences = []
    for key in v5_by_key:
        selected = _strip_parent_fields(output_by_key[key])
        if selected != v5_by_key[key]:
            source_differences.append(key)
            assert key == (TARGET_ROLE, TARGET_SMILES)
            assert selected == v3_by_key[key]
        else:
            assert selected == v5_by_key[key]
    assert source_differences == [(TARGET_ROLE, TARGET_SMILES)]

    c16 = next(
        record
        for record in output_records
        if record["source_class"] == "exact_c16_four_step_route_two_current_terminals"
    )
    assert c16["route_complete"] is True
    assert c16["route_step_count"] == c16["exact_l2_steps"] == 4
    assert c16["terminal_availability"] == "current_closed"
    c18 = output_by_key[(TARGET_ROLE, TARGET_SMILES)]
    assert c18["assessment_outcome"] == "missing_knowledge"
    assert c18["route_complete"] is False
    assert c18["route_step_count"] == c18["exact_l2_steps"] == 0
    assert c18["source_class"] == "v3_exact_overlay_or_replay"


def test_parent_ledger_is_accepted_by_downstream_loader_without_building_pair() -> None:
    records = _load_parent_records(LEDGER, expected_schema=LEDGER_SCHEMA_VERSION)
    assert len(records) == 1919
    target = next(
        record
        for record in records
        if (record["role"], record["canonical_smiles"]) == (TARGET_ROLE, TARGET_SMILES)
    )
    assert target["route_complete"] is False
    assert target["family_template_admitted"] is False


def test_lineage_is_hash_bound_to_final_v5_and_immutable_v3() -> None:
    result = json.loads(RESULT.read_text())
    config = json.loads(CONFIG.read_text())
    ledger = _gzip_payload(LEDGER)
    assert result["lineage"] == ledger["lineage"]
    assert result["lineage"]["final_v5"]["component_ledger"]["sha256"] == sha256_file(V5_LEDGER)
    assert result["lineage"]["pre_c18_v3"]["component_ledger"]["sha256"] == sha256_file(V3_LEDGER)
    for source in ("final_v5", "pre_c18_v3"):
        for artifact in ("component_ledger", "result", "config"):
            record = result["lineage"][source][artifact]
            assert sha256_file(REPO / record["path"]) == record["sha256"]
    assert result["policy"] == config["policy"] == REQUIRED_POLICY


def test_no_family_homologue_pair_or_holdout_promotion() -> None:
    config = json.loads(CONFIG.read_text())
    result = json.loads(RESULT.read_text())
    records = _gzip_payload(LEDGER)["records"]
    assert all(record["family_template_admitted"] is False for record in records)
    assert all(record["homologue_template_admitted"] is False for record in records)
    assert result["summary"]["family_template_admitted_records"] == 0
    assert result["summary"]["homologue_template_admitted_records"] == 0
    assert result["summary"]["registry_pair_materialized"] is False
    assert result["summary"]["holdout_artifacts_accessed"] == 0
    assert result["adjudication"]["registry_pair_materialized"] is False
    assert result["adjudication"]["holdout_accessed"] is False
    assert result["adjudication"]["holdout_reveal_authorized"] is False
    assert config["decision"] == {
        "registry_pair_materialized": False,
        "holdout_accessed": False,
        "holdout_reveal_authorized": False,
    }
    assert not any("holdout" in label.lower() for label in config["inputs"])
    assert not any("holdout" in record["path"].lower() for record in config["inputs"].values())


def test_input_hash_change_fails_closed(tmp_path: Path) -> None:
    config = _changed_config(
        tmp_path,
        lambda value: value["inputs"]["v5_component_ledger"].update({"sha256": "0" * 64}),
    )
    with pytest.raises(
        Ugi3FrozenParentComponentLedgerError,
        match="input hash changed: v5_component_ledger",
    ):
        build_frozen_parent_component_ledger(REPO, config)


def test_c18_must_come_from_noncomplete_v3_record(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text())
    v3_payload = _gzip_payload(V3_LEDGER)
    v5_target = _by_key(_gzip_payload(V5_LEDGER)["records"])[(TARGET_ROLE, TARGET_SMILES)]
    for index, record in enumerate(v3_payload["records"]):
        if (record["role"], record["canonical_smiles"]) == (TARGET_ROLE, TARGET_SMILES):
            v3_payload["records"][index] = v5_target
            break
    changed_ledger = tmp_path / "changed_v3.json.gz"
    changed_ledger.write_bytes(_gzip_bytes(v3_payload))

    source_result_path = REPO / config["inputs"]["v3_result"]["path"]
    changed_result_payload = json.loads(source_result_path.read_text())
    changed_result_payload["artifacts"]["component_synthesis_values.json.gz"]["sha256"] = (
        sha256_file(changed_ledger)
    )
    changed_result = tmp_path / "changed_v3_result.json"
    changed_result.write_text(json.dumps(changed_result_payload, indent=2, sort_keys=True) + "\n")
    config["inputs"]["v3_component_ledger"] = {
        "path": str(changed_ledger),
        "sha256": sha256_file(changed_ledger),
    }
    config["inputs"]["v3_result"] = {
        "path": str(changed_result),
        "sha256": sha256_file(changed_result),
    }
    changed_config = tmp_path / "config.json"
    changed_config.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        Ugi3FrozenParentComponentLedgerError,
        match="C18 source records violate the frozen reversion",
    ):
        build_frozen_parent_component_ledger(REPO, changed_config)


def test_policy_cannot_authorize_promotion_or_holdout_access(tmp_path: Path) -> None:
    config = _changed_config(
        tmp_path,
        lambda value: value["policy"].update(
            {
                "permit_family_template_promotion": True,
                "permit_homologue_template_promotion": True,
                "permit_registry_pair_materialization": True,
                "permit_holdout_access_or_reveal": True,
            }
        ),
    )
    with pytest.raises(Ugi3FrozenParentComponentLedgerError, match="policy changed"):
        build_frozen_parent_component_ledger(REPO, config)
